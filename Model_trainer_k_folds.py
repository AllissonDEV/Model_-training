import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Para reduzir logs
# Se quiser salvar:
import json


# Configurações
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
KFOLDS = 5
DATASET_PATH = "Samples/Train"

# Preparar DataFrame com imagens e classes
image_paths = []
labels = []

for class_name in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)

df = pd.DataFrame({'filename': image_paths, 'class': labels})
class_names = sorted(df['class'].unique())
num_classes = len(class_names)

# Class indices
class_indices = {cls: i for i, cls in enumerate(class_names)}
df['class_idx'] = df['class'].map(class_indices)


# Criar gerador de dados
def create_generator(df_subset, is_training):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25 if is_training else 0,
        width_shift_range=0.2 if is_training else 0,
        height_shift_range=0.2 if is_training else 0,
        shear_range=0.2 if is_training else 0,
        zoom_range=0.2 if is_training else 0,
        horizontal_flip=is_training,
        fill_mode='nearest'
    )
    return datagen.flow_from_dataframe(
        df_subset,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        class_mode='categorical',
        shuffle=is_training,
        batch_size=BATCH_SIZE,
        seed=SEED
    )


# Criar modelo com EfficientNetB3
def create_model():
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
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


# K-Fold Cross Validation
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
histories = []
fold = 1

for train_idx, val_idx in skf.split(df['filename'], df['class_idx']):
    print(f"\n--- Treinando Fold {fold} ---")
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Geradores
    train_gen = create_generator(train_df, is_training=True)
    val_gen = create_generator(val_df, is_training=False)

    # Pesos das classes


    classes_unique = np.unique(train_df['class_idx'])
    weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=train_df['class_idx'])
    weights = weights.tolist()  # <== converte para lista de floats nativos
    class_weights = dict(zip(classes_unique, weights))

    model = create_model()

    callbacks = [
        ModelCheckpoint(f"efficient_fold{fold}.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )


    def sanitize_history(history_dict):
        sanitized = {}
        for key, value in history_dict.items():
            sanitized[key] = [float(v.numpy()) if hasattr(v, 'numpy') else float(v) for v in value]
        return sanitized


    # Uso após o treino
    history_clean = sanitize_history(history.history)



    with open('history.json', 'w') as f:
        json.dump(history_clean, f, indent=4)

    histories.append(history)
    fold += 1

# Plotagem das curvas de acurácia por fold
plt.figure(figsize=(14, 6))
for i, h in enumerate(histories):
    plt.plot(h.history['val_accuracy'], label=f'Fold {i + 1}')
plt.title('Validação - Acurácia por Fold')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()
