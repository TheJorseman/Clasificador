from keras.models import load_model
from keras.preprocessing import image
import numpy as np
img_width, img_height = 150, 150
# load model
model = load_model('classificator.h5')
# Resumen del modelo
#model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# predicting images
file_name = 'test_not1.jpg'
img = image.load_img(file_name, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes[0][0])
if classes[0][0] == 1 :
    print( file_name + "\t Es un semaforo")
else:
    print(file_name + "\t No es un semaforo")
