# ConvolutionalNeuralNetwork-TF

## Deep MNIST CNN without Pooling
### DeepMnistCnnNoPooling.py
Convolutional Nerual Network without Pooling.
* 3 Convolutional layers
* 2 Fully Connected Layers

Adds **non-linearity** after each of the above layers.

[conv -> relu -> conv -> relu -> conv -> relu -> fc -> relu -fc -> relu]

This model does not use pooling to reduce the size. 
The **3rd hidden layer** (conv3) uses a **stride** of **2** to reduce the size.

Not using pooling is an idea seen here:[ Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)

Accuracy: **99.1**

### Notes
These **convolutional** layers do not use **zero-padding** so
the sizes of the output volume changes. This means that I had to carefully
track the output volumes.

The output volumes can be calculated with
#### ((W - F + 2P)/ (S)) + 1
* W = Input Volume size
* F = Filter Size ("receptive field size")
* P = Zero Padding used
* S = Stride

Because we use 0 zero-padding it can be simplified to:
#### ((W - F)/(S)) + 1

Another **disadvantage** to not using **zero-padding** is that 
the edges at each convolutional don't get **as many neurons looking** at them
compared to a traditional architecture with **zero-padding** at every conv layer 
to keep the volume the same( usually reduced by max-pooling).