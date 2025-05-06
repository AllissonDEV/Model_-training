
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import cv2

# Caminho para os dados
data_dir = r'Samples'

# Parâmetros do modelo
img_height, img_width = 224, 224
batch_size = 16
epochs = 15
num_folds = 3

# Carregamento manual das imagens e labels
X = []
y = []

for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_width, img_height))
                img = img.astype('float32') / 255.0
                X.append(img)
                y.append(class_folder)
            except:
                continue

X = np.array(X)
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Separa 10% para teste final
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# K-Fold nos 90% restantes
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

best_accuracy = 0.0
best_fold = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp), 1):
    print(f"\n🔁 Treinando Fold {fold}...")

    X_train, X_val = X_temp[train_idx], X_temp[val_idx]
    y_train, y_val = y_temp[train_idx], y_temp[val_idx]

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(y.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"✅ Fold {fold} - Val Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_fold = fold
        model.save(f'best_model_fold_{fold}_acc_{val_accuracy:.4f}.h5')

# Avaliação no conjunto de teste final
print("\n🏁 Avaliando no conjunto de TESTE FINAL...")

best_model = load_model(f'best_model_fold_{best_fold}_acc_{best_accuracy:.4f}.h5')
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Test Loss: {test_loss:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
