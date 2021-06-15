import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from classification.config import *

from classification.models.siamese_network.siamese_network import build_siamese_model
from classification.models.siamese_network.utils import *

def initialize_siamese_network():
    """
    Initilizes custom feature extracting CNN
    Compiles Siamese Network by joining two CNNs
    Sets training hyperparameters
 
    
    Returns:
    model, tf-keras compiled model ready for fitting
    """
    # configure the siamese network
    imgA = Input(shape=IMG_SHAPE)
    imgB = Input(shape=IMG_SHAPE)
    featureExtractor = build_siamese_model(IMG_SHAPE)
    # since sister networks, we buiild two instances of the SAME class
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    # final layers of the siamese network
    distance = Lambda(euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)

    # compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam",
        metrics=["accuracy"])
    
    return model


def initialize_triple_siamese_network():
    """
    Initilizes custom feature extracting CNN
    Compiles Triple Siamese Network by joining two CNNs
    Sets training hyperparameters
 
    
    Returns:
    model, tf-keras compiled model ready for fitting
    """
    # configure the siamese network
    featureExtractor = build_siamese_model(IMG_SHAPE)

    imgAnchor = Input(shape=IMG_SHAPE, name='anchor_input')
    imgPos = Input(shape=IMG_SHAPE, name='positive_input')
    imgNeg = Input(shape=IMG_SHAPE, name='negative_input')

    # since sister networks, we buiild two instances of the SAME class
    featsAnchor = featureExtractor(imgAnchor)
    featsPos = featureExtractor(imgPos)
    featsNeg = featureExtractor(imgNeg)

    # set Adam optimizer for fitting
    adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    # final layers of the siamese network
    merged_vector = concatenate([featsAnchor, featsPos, featsNeg], axis=-1, name='merged_layer')

    model = Model(inputs=[imgAnchor, imgPos, imgNeg], outputs=merged_vector)
    model.compile(loss=triplet_loss, optimizer=adam_optim)
    
    return model


def set_callbacks(CALLBACK_FLAGS):
    """
    Creates training callbacks, such as ModelCheckpoint and EarlyStopping
    
    Note for future implementation:
    Add TensorBoard callback
 
    Returns:
    CALLBACKS, array of model training callbacks, used when fitting model as a hyperparameter
    """
    CALLBACKS = []
    (EARLY_STOPPING_FLAG, MODEL_CHECKPOINT_FLAG) = CALLBACK_FLAGS 
    
    if EARLY_STOPPING_FLAG:
        # model callbacks
        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        CALLBACKS.append(es)
    
    return CALLBACKS


def initialize_resnet_tsnn():
    """
    Initilizes ResNet101 feature extracting CNN
    Compiles Triple Siamese Network by joining two CNNs
    Sets training hyperparameters
 
    
    Returns:
    model, tf-keras compiled model ready for fitting
    """
     # configure the siamese network
    featureExtractor = build_resnet101(IMG_SHAPE)
    
    imgAnchor = Input(shape=IMG_SHAPE, name='anchor_input')
    imgPos = Input(shape=IMG_SHAPE, name='positive_input')
    imgNeg = Input(shape=IMG_SHAPE, name='negative_input')

    # since sister networks, we buiild two instances of the SAME class
    featsAnchor = featureExtractor(imgAnchor)
    featsPos = featureExtractor(imgPos)
    featsNeg = featureExtractor(imgNeg)

    # set Adam optimizer for fitting
    adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    # final layers of the siamese network
    merged_vector = concatenate([featsAnchor, featsPos, featsNeg], axis=-1, name='merged_layer')

    model = Model(inputs=[imgAnchor, imgPos, imgNeg], outputs=merged_vector)
    model.compile(loss=triplet_loss, optimizer=adam_optim)
    
    return model

    
    