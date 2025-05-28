import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns

# Diretórios
DATA_SET_DIR = 'Samples'
TRAIN_DIR = 'Samples/Train'
VALIDATION_DIR = 'Samples/Test'

# Hiperparâmetros
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50

# Generators de imagem com aumento de dados
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Geradores de dados
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED,
        subset='training'
    )

    validation_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED
    )

    num_train_samples = train_generator.samples
    num_validation_samples = validation_generator.samples
    num_test_samples = test_generator.samples

    classes = list(train_generator.class_indices.keys())
    num_classes = len(classes)

    print(f"Classes detectadas: {classes}")
    print(f"Número de classes: {num_classes}")
    print(f"Amostras de treino: {num_train_samples}")
    print(f"Amostras de validação: {num_validation_samples}")
    print(f"Amostras de teste: {num_test_samples}")

except Exception as e:
    print(f"Erro ao carregar os dados: {e}")

# Função para criar o modelo
def create_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Criar modelo
try:
    model = create_model()
    model.summary()
except Exception as e:
    print(f'Erro ao criar modelo: {e}')

# Callbacks
callbacks = [
    ModelCheckpoint(
        filepath='sequential_2.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
]

# Treinamento do modelo
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    print('Treinamento concluído com sucesso!')
except Exception as e:
    print(f"Erro no treinamento: {e}")

# Avaliação do modelo
try:
    best_model = load_model('sequential_2.keras')

    # Reset do generator antes de avaliar ou prever
    test_generator.reset()

    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f'Acurácia no conjunto de teste: {test_accuracy:.4f}')
    print(f'Loss no conjunto de teste: {test_loss:.4f}')
except Exception as e:
    print(f"Erro na avaliação do modelo: {e}")

# Gráficos de desempenho
try:
    plt.figure(figsize=(16, 6))

    # Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    # Perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Erro ao gerar gráficos: {e}")

# Matriz de confusão e relatório
try:
    test_generator.reset()

    Y_pred = best_model.predict(test_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(test_generator.classes, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.show()

    print("Relatório de Classificação:")
    print(classification_report(test_generator.classes, y_pred, target_names=classes))
except Exception as e:
    print(f"Erro ao gerar relatório de classificação: {e}")
