from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np

model = load_model('sequential_2.keras')
classes = ['A', 'B', 'C', 'D', 'E', ...]  # sua lista de classes
IMG_SIZE = (128, 128)

img_path = 'caminho/para/imagem.jpg'
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
label = classes[np.argmax(pred)]

print(f'Letra prevista: {label}')
