# styleTransfer

## Description

**styleTrans.py**:  image prepocessing and transfer model setting

**main.py**: integrated user interface

## Overview

**Through the total loss function composed of content loss function and style loss function, the generated image combines the characteristics of content image and style image to realize style transfer.**

**The value of the loss function is calculated by the Gram matrix after output from the model constructed by the convolution layer in the VGG-19 imported by tensorflow**

Run main.py

![image](https://user-images.githubusercontent.com/89956877/206197027-f8ef8ed6-a72b-4e20-9d76-8ba9a9fd3d8e.png)

PS:

a. Upper left is one of your input —— content image; Upper right is style image;  

b. Bottom left is your configuration of transfer model and display of training epoch;  

c. Bottom right is the result image.  
