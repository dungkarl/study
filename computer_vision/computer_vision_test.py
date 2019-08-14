import tensorflow as tf 
from tensorflow.keras.applications import VGG16 


vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
        
vgg_conv.summary()