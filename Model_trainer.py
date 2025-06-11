# Imports das bibliotecas necessárias
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configurações
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 65

# Caminho dos dados (usaremos apenas o diretório de treino com split interno)
DATASET_PATH = "Samples"
TRAIN_DIR = os.path.join(DATASET_PATH, "Train")

# Geradores com aumento de dados
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2  # Usaremos esse split em vez de diretório separado para validação
)

# Geradores
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=SEED, subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False, seed=SEED, subset='validation'
)

# Classes e pesos
classes = list(train_generator.class_indices.keys())
num_classes = len(classes)
labels = np.unique(train_generator.classes)

weights = compute_class_weight(
    class_weight='balanced',
    classes=labels,
    y=train_generator.classes
)

class_weights = {int(label): float(w) for label, w in zip(labels, weights)}

# Criar modelo EfficientNetB3
def create_model():
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='efficientnetb3')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint('efficient_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
]

# Treinamento inicial
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# Fine-tuning
base_model = model.get_layer('efficientnetb3')
base_model.trainable = True

# Congele as camadas iniciais para não sobrecarregar o aprendizado
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=callbacks
)

# Avaliação final (usando o conjunto de validação separado internamente)
val_loss, val_acc = model.evaluate(validation_generator)
print(f"\nAcurácia final na validação: {val_acc:.4f}")

# Gráfico de acurácia da validação (antes e depois do fine-tuning)
def plot_history(histories, titles):
    plt.figure(figsize=(8, 5))
    for hist, label in zip(histories, titles):
        plt.plot(hist.history['val_accuracy'], label=label)
    plt.title("Evolução da Acurácia na Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_history([history, fine_tune_history], ["Pré-treino", "Fine-tuning"])
