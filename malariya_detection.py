# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:59:09 2020

@author: shalo
"""


# Loading the vgg19 Model
"""
Using vgg19 for our Malariya Detection

"""

#Freeze all layers except the top 4, as we'll only be training the top 4

import keras
from keras.applications import VGG19

#VGG19=keras.applications.vgg19.VGG19()#to check ALL the layers
#VGG19.summary()
#VGG19 was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

#FC=fully connected
# Re-loads the VGG19 model without the top or FC layers
# Here we freeze the last 4 layers 
VGG19 = VGG19(weights = 'imagenet', 
                 include_top = False,#not to train the top layer(or last layer because last layer is classifying 1000 classes but we want only two classes to be classified)
                 input_shape = (img_rows, img_cols, 3))

VGG19.summary()

# Layers which are set to trainable as True by default we make them false(not to train because it require high computational power and they are already trained)
for layer in VGG19.layers:
    layer.trainable = False
    
    
# Let's print our layers 
for (i,layer) in enumerate(VGG19.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


#Let's make a function that returns our FC Head

def addTopModelVGG19(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed on top of the last layers of VGG19"""

    top_model = bottom_model.output
    #top_model = GlobalAveragePooling2D()(top_model)
    top_model = Flatten()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)#activation='softmax' bcz we are using 2 nodes in the last layer otherwise it can be sigmoid also
    return top_model

#Let's add our FC Head back onto VGG19

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D#dropout=leave some perceptron of last layer(reduce overfitting )
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set our class number to 2 ('Parasite', 'Uninfected')
num_classes = 2

FC_Head = addTopModelVGG19(VGG19, num_classes)

model = Model(inputs = VGG19.input, outputs = FC_Head)#integrating both cnn vgg19 model and user model

print(model.summary())

#Loading our Malariya Dataset

from keras.preprocessing.image import ImageDataGenerator#for data augmentation


train_data_dir = r'C:\Users\shalo\Desktop\ML stuffs\projects\end2end\flask_DL\Dataset\Train'
validation_data_dir = r'C:\Users\shalo\Desktop\ML stuffs\projects\end2end\flask_DL\Dataset\Test'

# Let's use some data augmentation 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      zoom_range = 0.2,
      shear_range = 0.2,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)# we do not require data augmentation on test data
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32#will use 32 images at one time
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')#bcz activation='softmax' is used and we are using 2 nodes otherwise it can be binary_crossentropy
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


#Training out Model


from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("Malariya_TFL_MdlCHCKPNT1.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = adam(lr = 0.0001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 10000# images in training folder of both classes
nb_validation_samples = 2000# images in test folder of both classes

# We only train 3 EPOCHS 
epochs = 3
batch_size = 32

import time
start=time.time()

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)#// means it wont consider float value only int values will be considered


model.save('Malariya_TFL_Erlystop1.h5')


stop=time.time()

print("time taken ", (stop-start)," s")

import matplotlib.pyplot as plt

# from IPython.display import Inline
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print(train_generator.class_indices)#{'Parasite': 0, 'Uninfected': 1}

#Testing our classifer on some test images

from keras.models import load_model
import numpy as np
path=r"C:\Users\shalo\Desktop\ML stuffs\projects\end2end\flask_DL\Malariya_TFL_Erlystop1(83).h5"
classifier = load_model(path)


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

Malariya_dict = {"[0]": "Parasite", 
                      "[1]": "Uninfected"}

def draw_test(name, pred, im):
    check = Malariya_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 0, 0, 500, 550 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, check, (152, 60) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)
    
    
image_path=r"C:\Users\shalo\Desktop\ML stuffs\projects\end2end\flask_DL\Dataset\validation\Parasitized\3.png"
input_im=image_path
input_im=cv2.imread(input_im)
input_original = input_im.copy()
input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
input_im = input_im / 255.#part of image processing (we also did the same in data augmentation while training)
input_im = input_im.reshape(1,224,224,3) 

# Get Prediction
res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
res2=classifier.predict(input_im, 1, verbose = 0)
# Show image with predicted class
draw_test("Prediction", res, input_original) 
cv2.waitKey(0)

cv2.destroyAllWindows()    



#another method to test
from keras.models import load_model
import numpy as np
path=r"C:\Users\shalo\Desktop\ML stuffs\projects\end2end\flask_DL\Malariya_TFL_Erlystop1(83).h5"
classifier = load_model(path)
image_path=r"C:\Users\shalo\Desktop\ML stuffs\projects\end2end\flask_DL\Dataset\validation\Parasitized\3.png"
from tensorflow.keras.preprocessing import image

img = image.load_img(image_path, target_size=(224, 224))

# Preprocessing the image
x = image.img_to_array(img)
# x = np.true_divide(x, 255)
## Scaling
x=x/255#part of image processing (we also did the same in data augmentation while training)
x = np.expand_dims(x, axis=0)
   

# Be careful how your trained model deals with the input
# otherwise, it won't make correct prediction!
#x = preprocess_input(x)

preds = classifier.predict(x)
preds=np.argmax(preds, axis=1)
if preds==0:
    
    print("The Person is Infected With Pneumonia")
else:
    print("The Person is not Infected With Pneumonia")







