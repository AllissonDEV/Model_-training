
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
DATA_SET_DIR = 'Samples'
TRAIN_DIR = 'Samples/Train'
VALIDATION_DIR = 'Samples/Test'

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50

CLASS_NAMES = ...

# Generator para o treinamento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,           # rotação aleatória
    width_shift_range=0.1,       # Deslocamento horizontal
    height_shift_range=0.1,      # Deslocamento vertical
    shear_range=0.1,             # Cisalhamento
    zoom_range=0.1,              # zoom
    horizontal_flip=True,        # Flip horizontal
    fill_mode='nearest',         # Método de preenchimento
    validation_split=0.2         # 20% para validação
)



# Generator para a validação e teste
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Criação dos geradores de fluxo de dados
try:
    # Generator para parte de treinamento
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED,
        subset='training'
    )

    # Generator para parte de validação
    validation_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED,
        subset='validation'
    )

    # Generator para conjunto de teste
    test_generator = test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED
    )

    # Obter informações sobre os conjuntos de dados
    num_train_samples = train_generator.samples
    num_validation_samples = validation_generator.samples
    num_test_samples = test_generator.samples

    # Lista de classes (letras do alfabeto em Libras)
    classes = list(train_generator.class_indices.keys())
    num_classes = len(classes)

    print(f"Classes detectadas: {classes}")
    print(f"Número de classes: {num_classes}")
    print(f"Amostras de treino: {num_train_samples}")
    print(f"Amostras de validação: {num_validation_samples}")
    print(f"Amostras de teste: {num_test_samples}")

except Exception as e:
    print(f"Erro ao carregar os dados: {e}")





#função para criar o modelo
def create_model():

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Congela o backbone no início

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


try:
    model = create_model()
    model.summary()
except Exception as e:
    print(f'Erro ao criar modelo: {e}')





#callbacks para salvar o modelo ou parar se não melhorar

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


try:
    history = model.fit(
        train_generator,
        steps_per_epoch = num_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data = validation_generator,
        validation_steps = num_validation_samples // BATCH_SIZE,
        callbacks = callbacks,
        verbose = 1

    )

    print('Treinamento concluido')

    # Avaliação do modelo
    # Carregar o melhor modelo salvo
    best_model = load_model('sequential_2.keras')


    # Avaliar no conjunto de teste
    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f'Acurácia no conjunto de teste: {test_accuracy:.4f}')
    print(f'Loss no conjunto de teste: {test_loss}')

    # Visualizar métricas do treinamento
    plt.figure(figsize=(16, 6))

    # Plot da acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend()

    # Plot da perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Gerar matriz de confusão
    Y_pred = best_model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(test_generator.classes, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.show()

    # Imprimir relatório de classificação
    print("Relatório de Classificação:")
    print(classification_report(test_generator.classes, y_pred, target_names=classes))

except Exception as e:
    print(f"Erro durante o treinamento ou avaliação: {e}")