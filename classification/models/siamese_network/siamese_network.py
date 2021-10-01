# import the necessary packages
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

from classification.config import *


def build_siamese_model(inputShape, embeddingDim=48):
    """
    Set up of custom CNN feature extracting backbone
    Network is setup in 2 sets of Convolutional, ReLU, MaxPooling and Dropout layers
    Top layer performs Global Average Pooling followed by a Fully Connected Layer
    Network outputs embeddings from image of size = embeddingDim, defaults to 48
    Arguments:
        - inputShape: tuple, such that (img_height,img_width, channels)
    Returns:
        - model: keras::model
    """
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    # build the model
    model = Model(inputs, outputs)
    # return the model to the calling function
    return model


def build_resnet101(inputShape, embeddingDim=48):
    """
    Set up of ResNet101 architecture as feature extraction backbone
    
    Observations regarding input shape...
    
    inputShape must be tuple such that: (img_height,img_width, channels)
    input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
    
    Top set consists in Global Average Pooling followed by Dropout @ 70% and a Fully Connected Layer
    
    Network outputs embeddings from image of size = embeddingDim, defaults to 48
    
    """
  
    # will initialize network with random weights
    base_model = applications.ResNet101(weights= None, include_top=False, 
                                        input_shape= inputShape)
    
    # grab backbone output and global avg pool it, major dropout rate 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    # generate n = embeddingDim embeddings out of every image to calculate distance  
    outputs = Dense(embeddingDim)(x)
    model = Model(inputs = base_model.input, outputs = outputs)
    
    # return the model to the calling function
    return model

