import os

from classification.dataset.utils import triplet_data_processing
from classification.models.siamese_network.utils import plot_training, plot_error_diff
from classification.models.siamese_network.train import initialize_triple_siamese_network, set_callbacks
from classification.config import *     
        
# read data from DATASET_PATH, preprocesses, split it accordingly and make image data pairs
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("STEP 1: Reading and preprocessing image dataset...") 
tupleTrain, labelTrain, tupleTest, labelTest = triplet_data_processing(DATASET_PATH, SPLIT_PER)

# initialize siamese network 
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("STEP 2: Compiling siamese network ...")
model = initialize_triple_siamese_network()

# train the model
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("STEP 3: Siamese network training will begin in any moment...")
CALLBACKS = set_callbacks(CALLBACK_FLAGS)
    
print(" ")
Anchor = tupleTrain[:,0,:]
Positive = tupleTrain[:,1,:]
Negative = tupleTrain[:,2,:]
Anchor_test = tupleTest[:,0,:]
Positive_test = tupleTest[:,1,:]
Negative_test = tupleTest[:,2,:]

history = model.fit([Anchor,Positive,Negative],y=labelTrain,
          validation_data=([Anchor_test,Positive_test,Negative_test], labelTest), 
          batch_size=BATCH_SIZE, epochs=EPOCHS,
         callbacks=CALLBACKS)


# serialize the model to disk
print(" ")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("STEP 4: Saving trained model and plotting model history...")
if not os.path.exists(BASE_OUTPUT):
    os.mkdir(BASE_OUTPUT)
model.save(MODEL_PATH, save_format="h5")

plot_training(history, PLOT_PATH, acc_flag=False)
plot_error_diff(history, BASE_OUTPUT)
