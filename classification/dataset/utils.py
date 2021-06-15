import os

import cv2
# generates a montage of images to visually validate that our pair generation process is working correctly
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from classification.dataset.augmentation import augment_data


def make_pairs(images, labels):
    """
    Generates model input image pairs
    
    Arguments:
    images: set of input images
    labels: corresponding set of image labels
    
    Returns:
    tuple of image pairs and corresponding image pair label 
    image pair labels indicate whether image pair is positive or negative
    """
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


def make_triplets(images, labels):
    """
    Generates model input image triplets, these are used when 
    training a Triple Siamese Network Model
    
    Arguments:
    images: set of input images
    labels: corresponding set of image labels
    
    Returns:
    tuple of image triplets and corresponding image triplet label 
    image triplet labels indicate whether image pair is positive or negative
    """
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    tripletImages = []
    tripletLabels = []
    
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
        
        try :
            # positive example pick
            # randomly pick an image that belongs to the *same* class label
            idxB = np.random.choice(idx[label])
            posImage = images[idxB]

            # negative example pick
            # create negative index list and randomly pick an image corresponding
            # to a label *not* equal to the current label
            negIdx = np.where(labels != label)[0]
            negImage = images[np.random.choice(negIdx)]

            # prepare a anchor, positive and negative tuple and update the images and labels lists, respectively
            tripletImages.append([currentImage, posImage, negImage])
            # append a 1 since first is always positive and 0 for negative image
            tripletLabels.append([1, 0])
            
        except:
            continue
        
    # return a 3-tuple of our image pairs and labels
    return (np.array(tripletImages), np.array(tripletLabels))


def import_dataset(dataset_path):
    """
    Imports and pre processes input image data
    Transforms every image to default image shape: (224, 224, 3) as VGG
    
    Arguments:
    dataset_path: local path to input images, these path must contain images ordered in the following fashion:
    dataset
       |- class_1
           |- example_1
           |- example_2
           ...
           |- example_
           |- example_m
       |- class_2
       ...
       |- class_n
    
    Returns:
    data_paths, array containing every single path of every input image
    images, array, set of input images
    labels, array, corresponding set of image labels
    data_df, Pandas DataFrame containing images and corresponding labels
    """
    labels = []
    product_names = []
    images = []
    data_paths = []
    data_df = []
    heights = []
    widths = []

    # dataset directory must have a folder per image class
    for product in os.listdir(dataset_path):

        product_path = os.path.join(dataset_path, product)

        nsamples = len(os.listdir(product_path))
        product_names.append(product)
        # now iterate over every sample for every class
        for sample in os.listdir(product_path):

            sample_path = os.path.join(product_path, sample)

            sample_img = cv2.imread(sample_path)
            # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

            (h, w) = sample_img.shape[:2]
            heights.append(h)
            widths.append(w)

            sample_img = cv2.resize(sample_img, (224, 224))

            data_paths.append(sample_path)
            images.append(sample_img)
            labels.append(product)
            data_df.append([product, sample_img])
            
    return data_paths, images, labels, data_df


def data_processing(DATASET_PATH, SPLIT_PER, aug_flag=False):
    """
    Imports data, splits into training and validation sets, 
    generates input image pairs
    
    Arguments:
    DATASET_PATH: path to input images
    SPLIT_PER: training split percentage from full dataset
    aug_flag: flag that indicates whether or not image augmentation should be performed
    
    Returns:
    pairTrain, set of training image pairs
    labelTrain, corresponding set of image pair labels 
    pairTest, set of validation image pairs
    labelTest, corresponding set of image pair labels
    """
    data_paths, images, labels, data_df = import_dataset(DATASET_PATH)
    n_real = len(labels) 
    
    # if augmentation is flagged, perform it
    if aug_flag:
        images, labels = augment_data(images, labels, 5)
        n_aug = len(labels)

        print("######### Augmented data from {} to {} images ##############".format(n_real, n_aug))
        print(" ")
    
    print("######### PROCESSING DATA ##############")
    # transform categorical classes to numerical values
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    print(" ")
    print("Labels to train a product classification model: ", list(le.classes_))
    dummy_labels = le.transform(labels)

    percentage = SPLIT_PER

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        dummy_labels,
        test_size=(1 - percentage),
        shuffle=True,
        random_state=42,
    )
    
    print(" ")
    print("Out of {} total samples, we've got {} training samples and {} test samples...".format(len(images), len(X_train), len(X_test)))


# Now that we've got our data loaded and split we may now build our positive and negative image pairs
    (pairTrain, labelTrain) = make_pairs(X_train, y_train)
    (pairTest, labelTest) = make_pairs(X_test, y_test)
    print("From which, {} training image pairs and {} test pairs were generated...".format(len(labelTrain), len(labelTest)))

    # We may now populate our image list for visualization... Basically we will just horizontaly stack image pairs in a canvas and put a text on said canvas depending on whether the pair is positive or not
    # initialize the list of images that will be used when building our montage
    images = []

    # loop over a sample of our training pairs
    for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):

        # grab the current image pair and label
        imageA = pairTrain[i][0]
        imageB = pairTrain[i][1]
        label = labelTrain[i]

        # to make it easier to visualize the pairs and their positive or
        # negative annotations, we're going to "pad" the pair with four
        # pixels along the top, bottom, and right borders, respectively
        # image pair image is 96,192,3 shape
        pair = np.hstack([imageA, imageB])
        (h, w) = pair.shape[:2]
        output = np.zeros((h + 8, w + 8, 3), dtype="uint8")
        output[4:h+4, 0:w] = pair

        # set the text label for the pair along with what color we are
        # going to draw the pair in (green for a "positive" pair and
        # red for a "negative" pair)
        text = "neg" if label[0] == 0 else "pos"
        color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)

        # resize canvas to 96x51 (so we can better see it), and then
        # draw what type of pair it is on the image
        vis = cv2.resize(output, (96, 51), interpolation=cv2.INTER_LINEAR)
        cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1)
        # add the pair visualization to our list of output images
        images.append(vis)

        # with out image stack done we build the montage for the images
        montage = build_montages(images, (96, 51), (7, 7))[0]
        
    # show the output montage
    cv2.imshow("Image pair montage obtained from training set", montage)
    cv2.waitKey(0)
    
    return pairTrain, labelTrain, pairTest, labelTest


def triplet_data_processing(DATASET_PATH, SPLIT_PER):
    """
    Imports data, splits into training and validation sets, 
    generates input image triplets, use when training a 
    Triple Siamese Network
    
    Arguments:
    DATASET_PATH: path to input images
    SPLIT_PER: training split percentage from full dataset
    aug_flag: flag that indicates whether or not image augmentation should be performed
    
    Returns:
    tupleTrain, set of training image pairs
    labelTrain, corresponding set of image pair labels 
    tupleTest, set of validation image pairs
    labelTest, corresponding set of image pair labels
    """
    
    data_paths, images, labels, data_df = import_dataset(DATASET_PATH)
    n_real = len(labels) 
    
    #images, labels = augment_data(images, labels, 5)
    #n_aug = len(labels)
    print("######### Total images processed: {} ##############".format(n_real))
    #print("######### Augmented data from {} to {} images ##############".format(n_real, n_aug))
    print(" ")
    
    print("######### PROCESSING DATA ##############")
    # transform categorical classes to numerical values
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    print(" ")
    print("Labels to train a product classification model: ", list(le.classes_))
    dummy_labels = le.transform(labels)

    percentage = SPLIT_PER

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        dummy_labels,
        test_size=(1 - percentage),
        shuffle=True,
        random_state=42,
    )
    
    print(" ")
    print("Out of {} total samples, we've got {} training samples and {} test samples...".format(len(images), 
                                                                                                 len(X_train), len(X_test)))
    # build the anchor, positive and negative image triplets
    (tupleTrain, labelTrain) = make_triplets(X_train, y_train)
    (tupleTest, labelTest) = make_triplets(X_test, y_test)
    print("Training triplets generated: ", len(y_train), len(tupleTrain), tupleTrain.shape)
    print("Testing triplets generated: ", len(y_test), len(tupleTest), tupleTest.shape)

    # initialize the list of images that will be used when building our
    # montage
    images = []

    # loop over a sample of our training pairs
    for i in np.random.choice(np.arange(0, len(tupleTrain)), size=(49,)):

        # grab the current image pair and label
        imageA = tupleTrain[i][0]
        imageB = tupleTrain[i][1]
        imageC = tupleTrain[i][2]

        # to make it easier to visualize the pairs and their positive or
        # negative annotations, we're going to "pad" the pair with four
        # pixels along the top, bottom, and right borders, respectively
        # image pair image is 96,192,3 shape
        triplet = np.hstack([imageA, imageB, imageC])
        (h, w) = triplet.shape[:2]
        output = np.zeros((h + 16, w + 16, 3), dtype="uint8")
        output[4:h+4, 0:w] = triplet

        # resize it from 60x36 to 96x51 (so we can better see it), and then
        # draw what type of pair it is on the image
        vis = cv2.resize(output, (96, 51), interpolation=cv2.INTER_LINEAR)

        # add the pair visualization to our list of output images
        images.append(vis)

        # with out image stack done we build the montage for the images
        montage = build_montages(images, (96, 51), (7, 7))[0]
        
    # show the output montage
    cv2.imwrite("montage.jpg", montage)
    # cv2.waitKey(0)
    
    return tupleTrain, labelTrain, tupleTest, labelTest
