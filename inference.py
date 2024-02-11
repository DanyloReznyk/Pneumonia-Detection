import os
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
import cv2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import *
from PIL import Image 
import gdown
from architectures import model1, model2, model3, model4
from constants import IMAGE_SIZE, ADJUSTED_IMAGE_SIZE, TEST_IMAGES_PATH, TRAIN_IMAGES_PATH, TRAINED_MODEL_PATH, SUBMISSION_DF_PATH
import argparse
from skimage.measure import label, regionprops
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run inference on images in a directory')
    parser.add_argument('directory', type=str, help='Path to the directory containing images')
    # Parse command-line arguments
    args = parser.parse_args()
    # Call inference function with the provided directory path
    print(args.directory)
    
def download_model(download=False):
    """
    Download the model from Google Drive if specified.

    Parameters:
    download (bool): Whether to download the model or not. Default is False.

    Returns:
    None
    """
    if download:
        file_id1 = '1N_VjndJMb07tV6C3ieiUUYwCbwPTuziE'
        destination1 = 'checkpoints/model1.h5'

        url1 = f'https://drive.google.com/uc?id={file_id1}'
        gdown.download(url1, destination1, quiet=False)

        file_id2 = '146iGpi6Fpykg0-yQ8LV_7Pb80dWH09jg'
        destination2 = 'checkpoints/model2.h5'
        
        url2 = f'https://drive.google.com/uc?id={file_id2}'
        gdown.download(url2, destination2, quiet=False)

        file_id3 = '1X2Ik8CyZYiAEv1jhh4ElxIq6yrsngj4R'
        destination3 = 'checkpoints/model3.h5'
        
        url3 = f'https://drive.google.com/uc?id={file_id3}'
        gdown.download(url3, destination3, quiet=False)

        file_id4 = '1ijOHD2Mr9Ojw9_W8kDRK1THog-xieBkX'
        destination4 = 'checkpoints/model4.h5'

        url4 = f'https://drive.google.com/uc?id={file_id4}'
        gdown.download(url4, destination4, quiet=False)


        print('Successfully downloaded model!')
# loads pre-trained model from pre-defined folder

def load_model(model_f, model_s, model_t, model_fo):
    """
    Load the pre-trained model.

    This function loads a pre-trained model from the specified checkpoint file.

    Returns:
    model: Loaded pre-trained model.
    """
    print('Loading pre-trained model.')

    # Instantiate the model
    model_1 = model_f()
    # Load weights from the checkpoint file
    model_1.load_weights('checkpoints/model1.h5')

    model_2 = model_s()
    model_2.load_weights('checkpoints/model2.h5')

    model_3 = model_t()
    model_3.load_weights('checkpoints/model3.h5')

    model_4 = model_fo()
    model_4.load_weights('checkpoints/model4.h5')

    print('Successfully loaded pre-trained model.')
    
    return model_1, model_2, model_3, model_4


def reduce_rectangle_size(rectangle, reduction_percentage=5):
    """
    Reduces the size of a rectangle by a certain percentage on each side.

    Parameters:
        rectangle (tuple): Tuple containing (x, y, width, height, confidence) of the rectangle.
        reduction_percentage (float): Percentage by which to reduce the size of the rectangle on each side.

    Returns:
        tuple: Tuple containing modified (x, y, width, height, confidence) of the rectangle.
    """
    x, y, width, height, conf = rectangle
    reduction_factor = reduction_percentage / 100
    reduce_width = int(width * reduction_factor)
    reduce_height = int(height * reduction_factor)
    new_x = x + reduce_width
    new_y = y + reduce_height
    new_width = width - 2 * reduce_width
    new_height = height - 2 * reduce_height
    return new_x, new_y, new_width, new_height, conf


def pixelwise_to_rectangle_mask(pixelwise_mask):
    """
    Convert pixelwise mask to rectangle mask.

    This function converts a pixelwise mask to a rectangle mask by thresholding,
    applying connected components, and extracting bounding box information.

    Parameters:
    pixelwise_mask (numpy.array): Pixelwise mask array.

    Returns:
    predictionString (str): String representation of rectangle masks.
    """
    # Threshold true mask
    comp = pixelwise_mask[:, :] > 0.5

    # Apply connected components
    comp = label(comp)

    predictionString = ''
    rectangles = []

    for region in regionprops(comp):
        y, x, y2, x2 = region.bbox
        height = y2 - y
        width = x2 - x
        area = height * width
        if area <= 25000:
            continue
        conf = np.mean(pixelwise_mask[y:y+height, x:x+width])
        rectangle = (x, y, width, height, conf)
        reduced_rectangle = reduce_rectangle_size(rectangle)
        x, y, width, height, conf = reduced_rectangle
        predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '

    return predictionString


def predict_all_images(model_1, model_2, model_3, model_4, predict_all=False):
    """
    Predict bounding box masks for all images in a directory using a trained model.

    Parameters:
    model: Trained model for prediction.
    predict_all (bool): Whether to predict masks for all images or not. Default is False.

    Returns:
    None
    """
    if predict_all:
        # Load submission DataFrame and image directory
        test_df = pd.read_csv(SUBMISSION_DF_PATH)
        test_images_dir = args.directory  # Assuming args is defined elsewhere
        
        # Get filenames of images in the directory
        filenames = os.listdir(test_images_dir)
        X = []

        # Iterate over images to prepare for prediction
        for index, patient_id in enumerate(filenames):
            image_path = os.path.join(test_images_dir, patient_id)
            img = pydicom.dcmread(image_path)
            img = img.pixel_array
            img = cv2.resize(img, (ADJUSTED_IMAGE_SIZE, ADJUSTED_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = preprocess_input(np.array(img, dtype=np.float32))
            X.append(img)

        X = np.array(X)
        
        print('Start predict')
        submission_dict = {}
        print('Predicting first model')
        pred1 = model_1.predict(X , verbose = 0)

        print('Predicting second model')
        pred2 = model_2.predict(X , verbose = 0)

        print('Predicting third model')
        pred3 = model_3.predict(X , verbose = 0)

        print('Predicting fourth model')
        pred4 = model_4.predict(X , verbose = 0)
        
        preds=np.array([pred1, pred2, pred3, pred4])


        weights = [0.25, 0.25, 0.25, 0.25]

        weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))

        preds = np.argmax(weighted_preds, axis=3)

        # Generate prediction strings for each image
        for pred, filename in zip(preds, filenames):
            pred = resize(pred, (1024, 1024), anti_aliasing=False, order=0)
            predictionString = pixelwise_to_rectangle_mask(pred)
            filename = filename.split('.')[0]
            submission_dict[filename] = predictionString

        # Create submission DataFrame and save to CSV
        seg_sub = pd.DataFrame.from_dict(submission_dict, orient='index')
        seg_sub.index.names = ['patientId']
        seg_sub.columns = ['PredictionString']
        seg_sub.to_csv('submission.csv')

        print('Saved prediction results in data/submission.csv')

#download_model(download = True)
model_1, model_2, model_3, model_4 = load_model(model1, model2, model3, model4)
predict_all_images(model_1, model_2, model_3, model_4, predict_all = True)