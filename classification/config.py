import os

# path to image dataset path
DATASET_PATH = os.path.join(
    "/home/jorge/PycharmProjects/vico/tracking/reidentification/siamese-networks-tf/base-repo-tf2/dataset/meatme/train/")

# training split, 1- SPLIT_PER = test data percentage
SPLIT_PER = 0.8

# CHOOSE MODEL TYPE
siamese_classification = True
# SIAMESE NETWORK HYPERPARAMETERS
# specify the shape of the inputs for our network
IMG_SHAPE = (224, 224, 3)
# dimension of embedding layer, i.e CNN (sister) output layer
EMBEDDING_DIM = 48
# specify the batch size and number of epochs
BATCH_SIZE = 16
EPOCHS = 2

EARLY_STOPPING_FLAG = True
MODEL_CHECKPOINT_FLAG = True
CALLBACK_FLAGS = [EARLY_STOPPING_FLAG, MODEL_CHECKPOINT_FLAG]

# define the path to the base output directory
BASE_OUTPUT = "classification/models/siamese_network/output/employee"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.join(BASE_OUTPUT, "tsnn_head")
PLOT_PATH = os.path.join(BASE_OUTPUT, "tsnn_head.png")
