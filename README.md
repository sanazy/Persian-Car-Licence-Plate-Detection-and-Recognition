# Persian-Car-Licence-Plate-Detection-and-Recognition
 
## YOLOV7  
To detect the location of the car license plate in the images, Yolov7 pre-trained model were used. Here is the short description of this high-performing model:

### Performance
Yolov7 was introduced in July 2022 and outperforms other object detection models in terms of accuracy and speed. In the image below, Yolov7 performance is compared to other well-known object detectors:

<p align="center">
 <img src="https://res.cloudinary.com/dyd911kmh/image/upload/v1665138395/YOLOV_7_VS_Competitors_4ad9ccaa6f.png" width="500"/>
</p>

### Architecture
Yolov7 model was trained on MS COOC dataset and no pre-trained weights were used for training 
\[[1](https://pythonistaplanet.com/yolov7/)\]. 

In Yolov7 architecture, Extended Efficient Layer Aggregation Network (E-ELAN) were used which leads the model to learn more features. Moreover, the architecture of Yolov7 has scaled by concatenating other models such as YOLO-R which can address different speeds for inference. Thanks to bag-of-freebies, Yolov7 has better accuracy without any increment in the inference speed
\[[2](https://www.datacamp.com/blog/yolo-object-detection-explained)\].

More information is available at this link \[[3](https://viso.ai/deep-learning/yolov7-guide/)\].

## Step 1: Automatic Number Plate Detection (ANPR) using Yolov7 pre-trained Model

### Step 1.1: Prepare the dataset 
In order to trainig, two car datasets are used which have annotations for licence plates. One of them is [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download) which consists of 433 images of licence plates. Another dataset is [IranianCarsNumberPlate](https://www.kaggle.com/datasets/skhalili/iraniancarnumberplate?resource=download) which has 442 images. The anotations of both dataset are in XML format.

To easily generate a uniform dataset out of the above mentioned datasets, [roboflow](https://roboflow.com/) platform is used. It automatically load the images with their corresponding annotations, manage train/val/test splits and also add preprocessing and augmentation steps to dataset. Finally, it gives a few lines of code which can easily integerated into the colab.  


### Step 1.2: Train the Yolov7 Model
In this step, `License-Plate-Detector.ipynb` is used. 

You might encounter following error when you want to train the yolov7 model:
`Indices should be either on cpu or on the same device as the indexed tensor`
To handle this bug, some minor changes to `loss.py` file as mentioned in this [link](https://stackoverflow.com/questions/74372636/indices-should-be-either-on-cpu-or-on-the-same-device-as-the-indexed-tensor) will help. 

Training is last for about 1 hour for 30 epochs. In the following image, the result of training is shown:

<p align="center">
 <img src="https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/blob/main/images/training_detector_result.png" width="700"/>
</p>

As it can be seen, the precision, recall and mAP@o.5 (mAP calculated at IOU threshold 0.5) of both training and validation data reaches around 0.9 through training time.

Here a few test images that their license plate are correctly predicted by trained model are shown. The model could correctly detect license plates and draw their corresponding bounding boxes in different circumstances with different angels and illumination. In addition, it could detect two or more license plates whenever more cars are available in the image:

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


## Step 2: Optical Character Recognition (OCR) using Image Processing Techniques and Convolutional Neural Networks (CNN)

In this step, `Train_CNN_Model_for_OCR.ipynb` is used.

After localizing the location of license plate in an image, now, it is time to recognize the characters written in the plate. 

### Step 2.1 Prepare the Dataset and Train the CNN model

To train an model to recognize the persian characters and numbers, we need to have a related dataset. After some reseach, a dataset named **Iranis** was found which is appropriate for training licence plate recognition applications. Iranis is a large-scale dataset consists of more than 83000 real-world images of persian characters and numbers of car license plates [?](https://arxiv.org/ftp/arxiv/papers/2101/2101.00295.pdf). 

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



### Step 2.2

## Step 3: Optical Character Recognition (OCR) using Yolov7 pre-trained Model

