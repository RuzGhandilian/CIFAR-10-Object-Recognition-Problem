# CIFAR-10 Classification with ResNet

The project shows a CNN application with ResNet architecture, which is used for image classification of CIFAR-10 dataset. CIFAR-10 is a basic dataset in computer vision containing 60,000 32x32 colored images representing 10 classes.

## Dataset

The CIFAR-10 dataset contains 60,000 images categorized into 10 classes: bird, cat, car, frog, deer, dog, airplane, horse, ship, and truck. The dataset of 50000 training images and 10000 test images is separated. In this project, we also divide the training set into two halves, one of which is 80% for training and the other one is 20% for validation.

## Model Architecture

The ResNet architecture put forward by He and et al. in the research paper "Deep Residual Learning for Image Recognition" is the one that we use. The residual connections included in the ResNet architecture are the reason for their success in training, as they are able to solve the problem of gradient vanishing when the training is conducted. The convolutional layers with batch normalization are given the form of a residual block and it is followed by a skip connection in the main structure of ResNet.

## Results

Epoch [10/10], Train Loss: 0.1014, Train Accuracy: 96.36%, Val Loss: 0.6924, Val Accuracy: 82.41%

## References

- CIFAR-10 Dataset. [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
