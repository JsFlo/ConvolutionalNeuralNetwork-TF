# ConvolutionalNeuralNetwork-TF

## Deep MNIST CNN without Pooling
### DeepMnistCnnNoPooling.py
Convolutional Nerual Network without Pooling.
* 3 Convolutional layers
* 2 Fully Connected Layers


This model does not use pooling to reduce the size. 
The **3rd hidden layer** (conv3) uses a **stride** of **2** to reduce the size.

Not using pooling is an idea seen here:[ Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)

Accuracy: **99.1**