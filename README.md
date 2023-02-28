# License Plate Recognition (LPR) of Persian Cars 
 
 
## Introduction

License plate recognition is the ability of the computer-based system to automatically get the image of cars, locate the position of license plate in it and then convert the optical data of plate into digital information. This technology can be used in several vehicle management applications such as ticketless parking, stolen vehicles detection, smart billing and many more \[[1](https://survisiongroup.com/post-what-is-license-plate-recognition)\].

Although there are many open-source projects which discuss how to implement such a system for engligh character license plates, few resources for persian license plates is availale. In addition, thanks to the advance of deep learning models, thesedays we can leverage complex and ease-of-use models such as yolo version 7 with higher accuracy for object detection tasks. 

In this project, the aim is to implement an end-to-end system that can detect perisan license plates of cars and recognize the characters and numbers inside it in an image. The project can be mainly splitted into two stages. First, detection of license plates of cars using yolo version 7, one of the latest object detection model. Second, segmentation and recognition of persian license plate characters. For the second stage, two different methods are used, one using image processing techniques and another one using yolo version 7, this time for the recognition of characters.


<p align="center">
 <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/project_overview.png" width="800"/>
</p>


Note: This project is mostly inspired by this helpful [link](https://www.youtube.com/watch?v=bgAUHS1Adzo).
 
## Step 0: Intro to YOLOV7  
To detect the location of the car license plate in the images, Yolov7 pre-trained model is used in this project. Here is the short description of this high-performing model:

### Performance
Yolov7 was introduced in July 2022 and outperforms other object detection models in terms of accuracy and speed. In the image below, Yolov7 performance is compared to other well-known object detectors:

<p align="center">
 <img src="https://res.cloudinary.com/dyd911kmh/image/upload/v1665138395/YOLOV_7_VS_Competitors_4ad9ccaa6f.png" width="500"/>
</p>

### Architecture
Yolov7 model was trained on MS COOC dataset and no pre-trained weights were used for training 
\[[2](https://pythonistaplanet.com/yolov7/)\]. 

In Yolov7 architecture, Extended Efficient Layer Aggregation Network (E-ELAN) were used which leads the model to learn more features. Moreover, the architecture of Yolov7 has scaled by concatenating other models such as YOLO-R which can address different speeds for inference. Thanks to bag-of-freebies, Yolov7 has better accuracy without any increment in the inference speed
\[[3](https://www.datacamp.com/blog/yolo-object-detection-explained)\].

More information is available at this [link](https://viso.ai/deep-learning/yolov7-guide/).


## Step 1: Number Plate Detection using Yolov7 pre-trained Model


### Step 1.1: Prepare the dataset 

In order to train the model, two car datasets are used which have annotations for licence plates. One of them is [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download) which consists of 433 images of licence plates. Another dataset is [IranianCarsNumberPlate](https://www.kaggle.com/datasets/skhalili/iraniancarnumberplate?resource=download) which has 442 images of persian cars. The anotations of both dataset are in XML format.

To easily generate a uniform dataset out of the above mentioned datasets, [roboflow](https://roboflow.com/) platform is used. It automatically loads the images with their corresponding annotations, manage train/val/test splits and also add preprocessing and augmentation steps to dataset. Finally, it gives a few lines of code which can easily integerated into the python code.  


### Step 1.2: Train the Yolov7 Model

Training is last for about 1 hour for 30 epochs. In the following image, the result of training is shown:

<p align="center">
 <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/training_detector_result.png" width="700"/>
</p>

As it can be seen, the precision, recall and mAP@0.5 (mAP calculated at IOU threshold 0.5) of both training and validation data reaches around 0.9 through training time.

Note: You might encounter following error when you want to train the yolov7 model:
`Indices should be either on cpu or on the same device as the indexed tensor`
To handle this bug, some minor changes to `loss.py` file as mentioned in this [link](https://stackoverflow.com/questions/74372636/indices-should-be-either-on-cpu-or-on-the-same-device-as-the-indexed-tensor) will help. 

Here a few test images that their license plate are correctly predicted by trained model are shown. The model could correctly detect license plates and draw their corresponding bounding boxes in different circumstances with different angles and illumination. In addition, it could detect two or more license plates whenever more cars are available in the image:

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/correct_plate_1.jpg" width="200" />
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/correct_plate_4.jpg" width="200" /> 
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/correct_plate_3.jpg" width="200" />
</p>

There are also some false positives where model incorrectly detect some rectangular shapes as license plate:

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/wrong_plate_1.jpg" width="200" />
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/wrong_plate_2.jpg" width="200" /> 
</p>


Note: For this step, `Train_Yolov7_for_LPD.ipynb` is used. 


## Step 2: Optical Character Recognition (OCR) 

After detecting the location of licence plate of car in an image, now, it is time to recognize the exact character and numbers written in the plate. For this step, many methods are described in the litriture. One of them is using open source packages like [Tesseract](https://github.com/tesseract-ocr/tesseract) and [EasyOCR](https://github.com/JaidedAI/EasyOCR) which support many languages including persian. Unfortunately, the problem with them is that they must be fine-tuned especially for the font used in persian license plates. Otherwise, they do not give acceptable result in recognizing the characters \[[4](https://haghiri75.com/2022/01/17/%D8%AE%D9%88%D8%A7%D9%86%D8%AF%D9%86-%D9%BE%D9%84%D8%A7%DA%A9-%D8%AE%D9%88%D8%AF%D8%B1%D9%88-%D8%A8%D8%A7-%DA%A9%D9%85%DA%A9-yolov5-%D9%88-%D9%BE%D8%A7%DB%8C%D8%AA%D9%88%D9%86/)\].

Another way for OCR of license plate is to find out where the location of each character/number is in the image by image processing techniques and then give it to a trained image classificatier model to distiguishe it correctley. Although it is a more acceptable method than abovemethoined ones, it needs a good collection of data for each character/number used in license plate for training. In addition, since image processing techniques are somehow a manual way of feature enginnering, it might have shortage to generalize rules for all circemstances. Despite the limits, this method is implemented in step 2.1 and results are discussed.

Another method is to train an object detection model such as yolov7, this time, for recognizing characters/numbers in license plates of cars. This method is implemented and discussed in step 2.2. 


### Step 2.1: OCR using Image Processing Techniques and Convolutional Neural Networks (CNN)

#### Step 2.1.1 Prepare the Dataset and Train the CNN model

To train a model to recognize the persian characters and numbers, we need to have a related dataset. After some reseach, a dataset named **Iranis** was found which is appropriate for training licence plate recognition applications. Iranis is a large-scale dataset consists of more than 83000 real-world images of persian characters and numbers of car license plates \[[5](https://arxiv.org/ftp/arxiv/papers/2101/2101.00295.pdf)\]. 

In the image below, some sample images of this dataset for each character and number is shown:

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/iranis_samples.png" width=500/>
</p>

As it can be seen, there are 28 classes in this dataset. It is good to mention that license plates with red, green and yellow backgrounds are related to govermental, police and pubic transportation and taxi cars. While the white background license plates are refered to the normal private cars. 


For training a nueral network, a dataset should be splitted into training, validation and test sets. Working with raw dataset of Iranis would not be helpful. Thus, a python package named [split-folders](https://pypi.org/project/split-folders/) is used to split a dataset into different abovementioned sets. In this way, each character and number have same ratio in training, validation and test sets.   

In order to recognize each character/number, first, we should train a image classification model on our prepared dataset. A roughly simple convolutional neural network, which its architecute is shown in the image below and drawn by [this](https://alexlenail.me/NN-SVG/AlexNet.html) website, is used for classification task:  

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/cnn_model_architecture.png" width=700/>
</p>

After training the CNN model for about 10 epochs, the validation and training loss becomes very small. In the graphs below, the accuracy and loss for training and validation in each epoch is shown:

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/cnn_model_result.png" width=700/>
</p>

Moreover, the confusion matrix shows that number of true positives for each class is much higher than false predictions as the diagonal valus are show higher numbers: 

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/cnn_confusion_matrix.png" width=500/>
</p>

Note: For this step, `Train_CNN_Model_for_LPR.ipynb` is used.


#### Step 2.2.2: Apply Image Processing Techniques  

After training a model to classify each character/number, it is time to segment each one and feed it to the model to recognize which character it is. In this section, several image processing techniques have been used for segmentation of characters in detected plate. 


In the following, the general steps which are taken for character segmentation and recognition are listed:

1) Detect the location of plate and change the detected plate to gray image
3) Find the longest line in the image and calculate the angle of longest line in respect to the image
5) Rotate the image if the calculated angle is greater than 10 or smaller than -10; in this way later classification problem will be easier \[[6](https://github.com/mrymsadeghi/Colab_notebooks/blob/main/plate_OCR_comparison.ipynb)\]
6) Preprocess the image including apply gaussian blur, otsu thereshold, dilation to make characters in plate recognizeble from noise and background, then find connected contours \[[7](http://dangminhthang.com/computer-vision/characters-segmentation-and-recognition-for-vehicle-license-plate/)\]
7) Search through found connected white blobs and check some criteria such as ratio and area; if ratio of hight to the width of rectangle containing the blob is greater than 0.8 and area of rectangle is greater than 100 then apply the image classification to recognize which character or number it is based on our trained ocr model \[[8](https://github.com/theAIGuysCode/yolov4-custom-functions)\]
8) Print the number of license plate and save the annotated image into the drive


In the image below, the result of some of the abovementioned steps for correctly recognized license plates are shown: 

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/ocr_image_processing_table.png" width=900/>
</p>


Although, image processing will get good results for some cases, it suffers from manual tuning of some manual parameters such as size of filters or ratio and area which discussed earlier. In addition, some persian characters such as B (ب) which has dots on its own, cannot be recognized as a whole character and their dots can be incorrectly classified as 0 (۰). Moreover, sometimes after theresholding, some noises are big enough that considered as white blobs and feed into the model for classification which leads to wrong detected number plates. 


Note: For this step, `LPR_Image_Processing.ipynb` is used.


### Step 2.2: OCR using Yolov7 pre-trained Model

In this step, instead of using image processing techniques, a new yolov7 model is used for object detection of characters inside license plate. 

#### Step 2.2.1: Prepare dataset  

First, we need an appropriate dataset for training an object detection model to recognize the persian character and numbers available in car lince plates. 
Since no such dataset was found through research, ome of data available in [Iran-Vehicle-plate-dataset](https://www.kaggle.com/datasets/samyarr/iranvehicleplatedataset) and [IranianCarsNumberPlate](https://www.kaggle.com/datasets/skhalili/iraniancarnumberplate?resource=download) are annotated manually. 

For this, the plate regions of images of above datasets cropped and then feed into roboflow platform. After that, around 200 images are annotated, in a such a way that each character or number in the license plate determined by a bounding box and labeled to its corresponding class. There are 25 classes including numbers from 0 to 9 and characters repeated in private persian cars. After train and validation split, some augementation including adding noise, blurring and rotation is added to the data. In this way, the number of images reaches to around 560 images.

In the image below, a view of some of annotated images are shown:

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/some_of_annotated_plates.png" width=500/>
</p>

This annotated dataset is publicly available [here](https://universe.roboflow.com/sa-sa-d6awq/ocr-rzlyj/dataset/8).

Note: For this step, `Separate_Plates_from_Car_Images.ipynb` is used.


#### Step 2.2.2: Train a model using yolov7

Like training yolov7 for object detection of licence plates in the images, here, a model is trained for detecting the location of each character in license plate and classying it. The prepared dataset from previous step is fed into the model. The model is trained for 100 epochs for about 1 hour. The result for the last epoch is 0.968, 0.993, 0.991 and 0.661 for precision, recall, mAP@0.5 and mAP@.5:.95 respectively. 

In the image below, the performance of model during the training is shown. It is clear that the performance get improved during the time:


<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/training_recognition_result.png" width=700/>
</p>

In the image below, the prediction of model for some test cases is shown:

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/correctly_labeled_plates.png" width=500/>
</p>


Note: For this step, `Train_Yolov7_for_LPR.ipynb` is used.


#### Step 2.2.3: 

After training the model using yolov7, in this step, a pipeline is created for two main steps of the project. First, detecting the license plate and second recognizing the character and numbers of plate for test images. The code written for this part is mainly get from [`detect.py`](https://github.com/augmentedstartups/yolov7/blob/main/detect.py) file of yolov7 codes. 


In the image below, the result of some of the abovementioned steps for correctly recognized license plates are shown: 

<p align="center">
  <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/ocr_yolov7_table.png" width=900/>
</p>


After repeating the prediction on more test images, it is revealed that the performance of 


Note: For this step, `LPR-using-yolov7.ipynb` is used.


## Conclusion and Future Work


Last words:

- Annotate more data for OCR
- Gather more data as test set
- Train for more epochs and tune the hyperparameters to improve the performance of models
- Investigate missclassfied license plates for detection and characters/numbers for recognition
- Recognize the characters/numbers of license plates other than private cars such as govermental, police and public transportation
- Recognize the characters/numbers of free zone and temporary passing license plates
- Deploy the trained model on a web/mobile application and examine the performance on real-world images
- Extend the overall idea of detecting and recognizing license plates of images into videos for tracked cars

