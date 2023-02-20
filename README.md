# Persian-Car-Licence-Plate-Detection-and-Recognition
 
## YOLOV7  
To detect the location of the car license plate in the images, Yolov7 pre-trained model were used. Here is the short description of this high-performing model:

### Performance
Yolov7 was introduced in July 2022 and outperforms other object detection models in terms of accuracy and speed. In the image below, Yolov7 performance is compared to other well-known object detectors:

<p align="center">
 <img src="https://res.cloudinary.com/dyd911kmh/image/upload/v1665138395/YOLOV_7_VS_Competitors_4ad9ccaa6f.png" width="600" height="300"  />
</p>

### Architecture
Yolov7 model was trained on MS COOC dataset and no pre-trained weights were used for training 
\[[1](https://pythonistaplanet.com/yolov7/)\]. 

In Yolov7 architecture, Extended Efficient Layer Aggregation Network (E-ELAN) were used which leads the model to learn more features. Moreover, the architecture of Yolov7 has scaled by concatenating other models such as YOLO-R which can address different speeds for inference. Thanks to bag-of-freebies, Yolov7 has better accuracy without any increment in the inference speed
\[[2](https://www.datacamp.com/blog/yolo-object-detection-explained)\].

More information is available at this link \[[3](https://viso.ai/deep-learning/yolov7-guide/)\].

## Step 1: Automatic Number Plate Detection (ANPR) using Yolov7 pre-trained Model

### Step 1.1: 
Gather dataset in roboflow

### Step 1.2:
In this step, `License-Plate-Detector.ipynb` should be used. 

You might encounter following error when you want to train the yolov7 model:
Indices should be either on cpu or on the same device as the indexed tensor

To handle this bug, some minor changes mentioned in this [link](https://stackoverflow.com/questions/74372636/indices-should-be-either-on-cpu-or-on-the-same-device-as-the-indexed-tensor) will help. 

## Step 2: Optical Character Recognition (OCR) using Image Processing Techniques and Convolutional Neural Networks (CNN)

### Step 2.1

### Step 2.2

## Step 3: Optical Character Recognition (OCR) using Yolov7 pre-trained Model

