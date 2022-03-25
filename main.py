#from tensorflow.python.keras import Sequential
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Activation
from tensorflow.keras.layers import  Dense,BatchNormalization,Flatten,Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train= "/drive/My Drive/Colab Notebooks/birds_CNN/train/"
test= "/drive/My Drive/Colab Notebooks/birds_CNN/test"

img = load_img(train + "AMERICAN COOT/001.jpg")

img = img_to_array(img)
input_shape = img.shape
print(input_shape)


clas = glob(train + "/*")
out_num = len(clas)
print(out_num)

clas = glob(train+ '/*')
num_clas = len(clas)
print('number of class : ',num_clas)

img_size= 0
for each in clas:
    img_size += len(glob(each + '/*'))
print('image size :', img_size)


batch_size = 32
epochs = 200
n = 3
depth = (n*6)+2
def lr_schedule(epoch):
    lr = 1e-3
    if epoch >180:
        lr *= 0.5e-3
    if epoch > 160:
        lr *= 1e-3
    if epoch > 120:
        lr *= 1e-2
    if epoch > 80:
        lr *= 1e-1
    return lr


def resnet_layer(inputs,
                 num_filters = 16,
                 kernel_size = 3,
                 strides = 1,
                 activation = 'relu',
                 batch_normalization = True,
                 conv_first = True):

    conv = Conv2D(num_filters,
                  kernel_size = kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation('relu')(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation('relu')(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes = 3):

    if (depth-2)%9 !=0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    num_filters_in = 16
    num_res_block = int((depth-2)/9)

    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    for stage in range(5):
        for res_block in range(num_res_block):
            activation = 'relu'
            batch_normalization= True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in*4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in*2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             conv_first=False)
            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            x = tensorflow.python.keras.layers.add([x,y])


        num_filters_in = num_filters_out


    x =BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    y = Flatten()(x)
    outputs = Dense(num_clas,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs ,outputs=outputs)
    return model

def resnet_v1(input_shape,depth,num_class = 3):
    if(depth-2)%6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    num_filters = 16
    num_res_blocks = int((depth-2)/6)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs = inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2

            y = resnet_layer(inputs = x,
                             num_filters=num_filters,
                             strides= strides)
            y = resnet_layer(inputs = y,
                             num_filters= num_filters,
                             activation= None)

            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs = x,
                                 num_filters = num_filters,
                                 kernel_size=1,
                                 strides= strides,
                                 activation=None,
                                 batch_normalization=False)

            x = tensorflow.keras.layers.add([x,y])
            x = Activation('relu')(x)
        num_filters *=2

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs = inputs , outputs = outputs)
    return model

model = resnet_v2(input_shape=input_shape,depth=depth)

model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


model_type = 'ResNet%dv%d' % (depth, 2)
save_dir = os.path.join(os.getcwd(),'saved models')
model_name = 'covid_19%s_model.{epoch:03d}.h5' % model_type


filepath = os.path.join(save_dir, model_name)

import time

start_time = time.time()

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]


train_datagen = ImageDataGenerator(rescale= 1./255,
                                   rotation_range = 45)

test_datagen = ImageDataGenerator(rescale= 1./255,
                                  rotation_range = 45)

train_generator = train_datagen.flow_from_directory(train,
                                  target_size= img.shape[:2],
                                  batch_size= batch_size,
                                  color_mode= "rgb",
                                  class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(test,
                                 target_size=img.shape[:2],
                                 color_mode= "rgb",
                                 class_mode= "categorical")

hist = model.fit_generator(generator= train_generator,
                           epochs=30, verbose=1, workers=4,
                           validation_data= test_generator,
                           callbacks=callbacks,
                           use_multiprocessing=False,
                           validation_steps=15)


scores = model.evaluate(test , verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])