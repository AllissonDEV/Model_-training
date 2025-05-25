import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Configurações
img_size = 224
dataset_path = 'Samples'
epochs = 35
batch_size = 64
num_folds = 3
output_dir = 'resultados_folds'
os.makedirs(output_dir, exist_ok=True)

# Função para carregar imagens
def load_images(path):
    X, y = [], []
    classes = sorted(os.listdir(path))
    for label in classes:
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)

print("🔄 Carregando imagens...")
X, y = load_images(dataset_path)
X = X.astype('float32') / 255.0

# Codificação de rótulos
le = LabelEncoder()
y_encoded = le.fit_transform(y
y_categorical = to_categorical(y_encoded)
class_names = le.classes_
num_classes = y_categorical.shape[1]

# Separar Hold-Out (20%)
X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
)

# K-Fold nos 80% restantes
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
acc_per_fold = []
loss_per_fold = []
fold_no = 1

for train_index, val_index in kf.split(X_train_full, y_train_full):
    print(f"\n🧪 Treinando Fold {fold_no}...")

    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    y_train, y_val = y_train_full[train_index], y_train_full[val_index]

    # Modelo
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(img_size,img_size,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(f"{output_dir}/modelo_fold_{fold_no}.h5", save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    scores = model.evaluate(X_val, y_val, verbose=0)
    print(f"✅ Fold {fold_no} - Acurácia: {scores[1]*100:.2f}%")
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_val, axis=1)

    report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4)
    with open(f"{output_dir}/relatorio_fold_{fold_no}.txt", "w") as f:
        f.write(report)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - Fold {fold_no}')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Perda - Fold {fold_no}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/grafico_fold_{fold_no}.png")
    plt.close()

    fold_no += 1

print("\n📊 Resultados por Fold:")
for i in range(len(acc_per_fold)):
    print(f"> Fold {i+1} - Acurácia: {acc_per_fold[i]:.2f}%, Perda: {loss_per_fold[i]:.4f}")

print(f"\n📈 Média Acurácia: {np.mean(acc_per_fold):.2f}%")
print(f"📉 Média Perda: {np.mean(loss_per_fold):.4f}")

# Treinamento Final com 80% dos dados
print("\n🧪 Treinando modelo final com TODO o conjunto de treino...")
final_model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(img_size,img_size,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

final_model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_final = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint(f"{output_dir}/modelo_final.h5", save_best_only=True, monitor='val_loss')
]

history_final = final_model.fit(
    X_train_full, y_train_full,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks_final,
    verbose=1
)

# Avaliação no Hold-Out
print("\n🧪 Avaliação final no conjunto Hold-Out...")
holdout_scores = final_model.evaluate(X_test_holdout, y_test_holdout, verbose=0)
print(f"🎯 Hold-Out - Acurácia: {holdout_scores[1]*100:.2f}%, Perda: {holdout_scores[0]:.4f}")

y_pred_holdout = final_model.predict(X_test_holdout)
y_pred_labels = np.argmax(y_pred_holdout, axis=1)
y_true_labels = np.argmax(y_test_holdout, axis=1)

report_holdout = classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4)
print(report_holdout)

with open(f"{output_dir}/relatorio_holdout.txt", "w") as f:
    f.write(report_holdout)
