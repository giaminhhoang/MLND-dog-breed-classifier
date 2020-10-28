[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Dog breed classifier is a representative image classification problem in the computer vision (CV). Classifying dogs into their breeds are often challenging even for human by simply looking at them as there are hundreds of distinct breeds. Thus, it needs to be done by computer instead. Solving this problem is useful when rescuing dogs, finding them homes, treating them, etc. or just simply seeing a cute dog and wondering which breed the dog belonged. Given an input image, the first task is to identify if it is a dog image or a human image. If it is a dog image then classify it to one breed of dog. If the image is a human face then identify the most resembling dog breed associated with the face. 

To solve the multi-class classification problem in CV, convolutional neural networks (CNNs) have been widely used for their efficiency to deal with 2-D data such as images. CNN can capture the spatial dependencies of pixels in an image by applying 2-D convolution filters. The architecture is suitable for image datasets since a small set of parameters (the kernel) is used to compute outputs of the entire image, so the model has much fewer parameters compared to a fully connected layer. Put differently, CNN can be trained to understand the sophistication of the image better than feedforward neural networks. Thus, CNN will be employed to accomplish this project. 

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
    
__NOTE:__ if you are using the Udacity workspace, you *DO NOT* need to re-download the datasets in steps 2 and 3 - they can be found in the `/data` folder as noted within the workspace Jupyter notebook.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure you have already installed the necessary Python packages according to the README in the program repository.
5. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```
## Algorithms and Techniques

To solve the multi-class classification problem in CV, CNNs have been widely adopted. CNN can capture the spatial dependencies of pixels in an image by applying 2-D convolution. The architecture is suitable for image datasets since a small set of parameters (the kernel) is used to compute outputs of the entire image, so the model has much fewer parameters compared to a fully connected layer. Put differently, CNN can be trained to understand the sophistication of the image better than feedforward neural networks. The solution involves 3 stages:

Human face detection using OpenCV's implementation of Haar feature-based cascade classifiers.

Dog detection using the VGG-16 model, along with weights that have been trained on ImageNet.

Dog breed classification using first a CNN built from scratch, then trying transfer learning with pre-trained models from ImageNet competition to significantly models boost the accuracy to meet the requirements of the project. Some good candidates are VGGNet and Residual Network (or ResNet). 

Besides, data augmentation is also adopted to extend a dataset and improve generalization to avoid overfitting. 

### Model Evaluation
<p align="justify">The CNN model created using transfer learning with
ResNet101 architecture was trained for 20 epochs, and the final model produced an
accuracy of 85% on test data. The model correctly predicted breeds for 718 images out of 836 total images.</p>

**Accuracy on test data: 85%**

