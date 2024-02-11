# Pneumonia-Detection
Data Mayhem solution's

# RSNA Pneumonia Detection Challenge using CNN

### Summary
This repository contains a project on image detection and semantic segmentation using pneumonia dataset.


- **Task:** Object Detection <br>
- **Data:** <a href='https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data'>Link</a> <br>
- **Neural Network:** U-Net like architectures using spatial attentions and ASPP
- **Scoring function:** Mean IoU and Accuracy

### How to run:
The inference.py file takes as an argument the path to the directory with RGB images:  
The code is run according to the following example:
python inference.py "DIRECTORY_TO_RGBIMAGES" or python inference.py input\stage_2_test_images

After that, thanks to the download_model function, model weights are downloaded to the checkpoints directory. Then, thanks to the load_model function, pretrained models are loaded.

After that starts predicting. 
In summary we have 4 models so because of that you need to wait a little during downloading and predicting process.

### Model results
- 0.1134 - private score
#### Model info
After many experiments, it was decided to use the following model: to use the segmentation model of the U-net architecture because in our opinion, segmentation is more appropriate here, since semantic segmentation provides pixel-level accuracy in labeling each pixel in the image with the corresponding object category. Such precise localization can improve the accuracy of object detection systems, especially in scenarios where objects are densely packed or overlap.
For this, 3 U-net architectures and 1 ASPP were used. An ensemble was then formed where all predictions from each model were weighted averaged.

More about each model:
The first is U-net with ResNet152 as backbone.
The second is U-net with DenseNet121 as backbone.
The third is U-net with VGG19 as backbone together with spatial attention for encoders and bridge.
The Fourth is ASPP model with ResNet152 as encoder.

Training was performed on 4800 images, which were stratified by class, and validation was performed on 1200 images. The batch size was 16 for training and 32 for validation, the learning rate was the same everywhere in 0.0001, the optimizer was adam. The following loss function was used: it is the weighted sum of IoU and binary crossentropy.
It is important to note that we reduced the image to 128 pixels, although this did not have ultra-high accuracy, but it gave speed, the number of channels - 3. The masks were transformed into a categorical array that stores a binary vector with a size of 2*1 for each pixel.

After predicting the test result, the expected masks were converted into 128*128*1 format and then, using the skimage method, pixels were found which, combined together, form a box that is a mask. At the same time, a certain post-processing is performed - all masks that are too small are discarded, only from 25,000 thousand pixels, and each mask is also reduced by 5 percent - this is explained by the fact that the model tends to slightly exaggerate the mask and draw small masks in unexpected places.

#### Model training history
<img src="results\f1-score.jpg"/>
<img src="results\loss.jpg"/>

#### Model evaluation
<img src="results\eval.jpg"/>

#### Prediction vs ground truth
<img src="results\validation\a8af12f5b.jpg"/>
<img src="results\validation\e5cb861f3.jpg"/>

#### Inference example
<img src="results\inference\3d75a5157.jpg"/>
<img src="results\inference\582ed5b82.jpg"/>
<img src="results\inference\d6cf01e6f.jpg"/>

### Project Structure:
```bash
├───checkpoints # Data folder for models weights
├───EDA.ipynb # EDA and initial data prep
├───architecture.py # All models
├───train.py # Define model
├───inference.py # Model inference 
├───submission.csv
├───checkpoints # Folder with best model checkpoints
├───constants.py # Declare variables
├───metrics.py # Metrics are used in training
└───requirements.txt
```
### Kaggle Notebook
Used it to run all model training <a href='https://www.kaggle.com/jeniagerasimov/airbus-semantic-segmantation'>View</a>

### Conclusion:
Project was very challenging as I had 0 previous experience with neural networks and keras, but it was very fun nonetheless.
