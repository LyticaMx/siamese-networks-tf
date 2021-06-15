# siamese-networks-tf


## How to train a custom feature extractor

Head over to `classification` and open up `config.py`, this is where youll find the training env variables and model hiperparameters that can be edited...

```
# path to image dataset path
DATASET_PATH = os.path.join("PATH/TO/WHERE/YOUR/DATASET/IS/LOCATED")
```
Data must be separated within directory in subdirectories, each representing each image class for which your feature extractor will be trained

Like this
```
dataset/
	class_1/
		sample_1.jpg
		.
		.
		.
		sample_m.jpg
	.
	.
	.
	class_n/
```
Then you can tune model hyperparameters like:
```
# training split, 1- SPLIT_PER = test data percentage
SPLIT_PER: PERCENTAGE OF SAMPLES TO BE USED FOR TRAINING

# CHOOSE MODEL TYPE
siamese_classification = True

# specify the shape of the inputs for our network
IMG_SHAPE = S-CNN INPUT SHAPE, e.g (224, 224, 3)
# dimension of embedding layer, i.e CNN (sister) output layer
EMBEDDING_DIM = NUMBER OF EMBEDDINGS AS OUTPUT, e.g. 48
# specify the batch size and number of epochs
BATCH_SIZE = NUMBER OF TRIPLETS USED PER BATCH, e.g. 128
EPOCHS = NUMBER OF TIMES WHOLE DATASET WILL BE FITTED, e.g. 100

EARLY_STOPPING_FLAG = STOP IF THERE HAS NOT BEEN ANY IMPROVEMENT IN TRAINING AFTER 5 EPOCHS, e.g True
MODEL_CHECKPOINT_FLAG = CREATE BEST MODEL CHECKPOINT, e.g. True
# ADD OTHER CUSTOM CALLBACKS IF FAMILIARIZED WITH TF
CALLBACK_FLAGS = [EARLY_STOPPING_FLAG, MODEL_CHECKPOINT_FLAG]
```
Then you must define where resulting model `.h5` file will be written at
```
# define the path to the base output directory
BASE_OUTPUT = "PATH/TO/OUTPUT/DIR"
# results
MODEL_PATH = os.path.join(BASE_OUTPUT, "MODEL_NAME")
PLOT_PATH = os.path.join(BASE_OUTPUT, "MODEL_TRAINING_PLOT.png")
```




