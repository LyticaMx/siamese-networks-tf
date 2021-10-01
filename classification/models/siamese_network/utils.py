# import the necessary packages
import os
import cv2
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


# copy paste of previously declared make_pairs() function
def make_pairs(images, labels):
    '''
    Create image negative and positive pairs to serve as input
    of siamese network model

    -- images: array that contains image dataset
    -- labels: array of corresponding image labels

    '''

    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    
    # get unique or total classes and set index list for each one of classes
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    
        # loop over all images
    # index A refers to current looped image, and B index refers to paired positive image
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        
        # positive pair generation
        # randomly pick an image that belongs to the *same* class label
        try :
            idxB = np.random.choice(idx[label])
            posImage = images[idxB]
            # prepare a positive pair and update the images and labels lists, respectively
            pairImages.append([currentImage, posImage])
            # 1 is appended since this pair is a positive image pair
            pairLabels.append([1])

            # negative pair generation
            # create negative index list and randomly pick an image corresponding
            # to a label *not* equal to the current label
            negIdx = np.where(labels != label)[0]
            negImage = images[np.random.choice(negIdx)]
            # prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            # 0 is appended since this pair is a negative image pair
            pairLabels.append([0])
        except:
            continue
        
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
    """
    Computes Euclidean distance between sister networks embedding outputs
    
    -- vectors: tuple of the outputs from each sister network, in this case a tuple containing two 48 dim arrays
    Returns
    distance: Euclidean distance between feature network arrays
    """
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    
    # return the euclidean distance between feature network arrays
    distance = K.sqrt(K.maximum(sumSquared, K.epsilon()))
    
    return distance

def triplet_loss(y_true, y_pred, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)
    
    total_embeddings = y_pred.shape.as_list()[-1]

    # from merged_vector we grab embeddings for each input image based on index
    featsAnchor = y_pred[:, 0:int(total_embeddings * 1/3)]
    featsPos = y_pred[:, int(total_embeddings * 1/3):int(total_embeddings * 2/3)]
    featsNeg = y_pred[:, int(total_embeddings * 2/3):int(total_embeddings)]

    # distance between the anchor and the positive
    pos_dist = euclidean_distance([featsAnchor, featsPos])

    # distance between the anchor and the negative
    neg_dist = euclidean_distance([featsAnchor, featsNeg])

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0)
 
    return loss


def plot_training(H, plotPath, acc_flag=True):
    """
    Generates training metric plots and writes plot image to disk 
    Arguments:
    
    H: refers to training history and plotPath to training history plot image saving path
    plotPath: local path where plot image will be written
    acc_flag: flag that indicates whether or not accuracy is being tracked in training
            
    Returns:
    plt figure, matplotlib plot of training statistics
    
    """

    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    
    if acc_flag:
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.plot(H.history["val_accuracy"], label="val_acc")
        
    plt.title("Training Metrics")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)
    # plt.show()
    
    

def plot_error_diff(H, base_output_path):
    """
    Generates plot that computes difference between training and validation error
    writes plot image to disk 
    Arguments:
    
    H: refers to training history and plotPath to training history plot image saving path
    plotPath: local path where plot image will be written
    acc_flag: flag that indicates whether or not accuracy is being tracked in training
            
    Returns:
    plt figure, matplotlib plot of training statistics
    
    """

    plotPath = os.path.join(base_output_path, "error_dif.png")
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    diff_error = list(np.array(H.history["loss"]) - np.array(H.history["val_loss"]))
    plt.plot(diff_error, label="error_diff")
    plt.title("Difference between training and test loss across training")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)
    # plt.show()
    
    
def eval_single_image(image, triple_model):
    """
    Pre processes and performs forward pass over single image
    in Sister CNN component of network
    The goal here is to obtain embeddings of image
    
    Arguments:
    image, input image, cvt::Mat object, numpy array
    triple_model, triple siamese network as model object from Keras
    
    Returns:
    embeddings, 48-d array of image embeddings, will be used to compute distance
        to other images
    """
    
    image = cv2.resize(image, (96, 96))
    # add a batch dimension to both images
    image = np.expand_dims(image, axis=0)
    # evaluate in sister CNN network, corresponds to 4 layer in Model
    # for more details, always check model.summary() and make sure you
    # understand what the model is actually doing
    embeddings = triple_model.layers[3](image)[0]

    return embeddings

def compare_outputs(featsAnchor, model_feats):
    """
    Compute distance between embeddings taken from anchor (input) image, 
    postive and negative images
    
    Arguments:
    featsAnchor: 48-d array of anchor image embeddings
    model_feats: set of image embeddings corresponding to each
        model images

    Returns:
    prediction: integer value from {0, ..., K} number of classes, 
    refers to class to which input image belongs
    distances: returns a list of distances calculated, where 
        distances[k] = distance from anchor image to K model image
    """
    
    distances = []
    # distance between the anchor and the positive
    #first_second_dist = euclidean_distance([featsAnchor, featsPos])
    for featsModel in model_feats:
        
        dist = np.linalg.norm(featsAnchor - featsModel)
        dist =  float("{:.4f}".format(dist))
        distances.append(dist)
    
    # if the distance of the anchor to the positive image is larger than that to the negative image
    # then anchor image must be alike to negative image
    prediction = np.argmin(distances)

 
    return prediction, distances

def load_model_images(model_image_dir):
    """
    Loads to memory input model images, and prepares them to be
    computed into embeddings
    
    Arguments:
    model_image_dir = string, path to model images directory 
    Note: name in images must reflect class to which each model
        image belongs

    Returns:
    model_images, list of model images
    labels, list of labels corresponding to model images
    """
    model_images = []
    labels = []
    
    images = os.listdir(model_image_dir)
    
    for image in images:
        
        image_path = os.path.join(model_image_dir, image)
        label = os.path.splitext(image)[0]
        
        image = cv2.imread(image_path)
        model_images.append(image)
        
        labels.append(label)
        
    return model_images, labels
    
    
def model_images_feats(model_images, triple_model):
    """
    Computes feature embeddings for every model image
    
    Arguments:
    model_images, list of model images
    triple_model, Keras trained Triple Siamese Network

    Returns:
    model_feats, list of model image embeddings
    """
    model_feats = []
    
    for image in model_images:
        
        feats = eval_single_image(image, triple_model)
        
        model_feats.append(feats)
        
    return model_feats
        
    

def classify_roi(triple_model, labels, roi_images, roi_bbxs, model_feats, draw_frame, colors):
    """
    Classifies every detected object obtained via
    Custom SKU Object detection, passes bounding boxes
    through Triple Siamese Network Model
    
    Arguments
    triple_model: Keras trained Triple SNN model
    labels: array of classification labels
    roi_images: array of input images, i.e. detections
    roi_bbxs: bounding boxes coordinates of input images
    model_feats: set of model images embeddings
    draw_frame: copy of original frame for detection drawing
    
    Returns
    class_preds: list of predicted class per detection
    draw_frame: copy of original frame drawn after detections
    """
    
    class_preds = []
    
    # loop over all image pairs
    for (i, (anchor_img, anchor_bbx)) in enumerate(zip(roi_images, roi_bbxs)):

        # load both the images and convert them to grayscale
        imageA = cv2.resize(anchor_img, (96, 96))

        # use our siamese model to make predictions on the image pair,
        # indicating whether or not the images belong to the same class
        # preds = triple_model.predict([imageA])
        featsAnchor = eval_single_image(imageA, triple_model)

        prediction, distances = compare_outputs(featsAnchor, model_feats)
        detected_class = labels[prediction]
        class_preds.append(detected_class)

        color = colors[prediction]
            
        xmin, ymin, xmax, ymax = anchor_bbx
            
        cv2.rectangle(draw_frame, (xmin, ymin), (xmax, ymax), color, 1)
            
    return class_preds, draw_frame
    