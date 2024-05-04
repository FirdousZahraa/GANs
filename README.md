# GANs 

# Introduction
This code trains a GAN (Generative Adversarial Network) to generate synthetic images resembling handwritten digits from the MNIST dataset. The generator learns to produce images that can fool the discriminator into classifying them as real, while the discriminator learns to better distinguish between real and fake images. This adversarial training process iteratively improves both networks until convergence.

## 1. Imports and Setup: 
Necessary libraries are imported, and parameters like CUDA usage, data paths, hyperparameters are set. These hyperparameters control various aspects of the training process, such as network architecture, optimization settings, and dataset handling. You can adjust these hyperparameters based on your specific requirements and computational resources.

## 2. Data Loading:
+ The MNIST dataset is loaded using `torchvision.datasets.MNIST`. This dataset contains grayscale images of handwritten digits along with their labels.
+ The dataset is transformed using `transforms.Compose` to resize images to a specified size, convert them to tensors, and normalize them.

## 3. Model Architecture:
+ Generator: The generator architecture consists of several convolutional transpose layers followed by batch normalization and activation functions (ReLU for hidden layers and Tanh for the output layer). It takes random noise as input and outputs synthetic images.
  
+ Discriminator: The discriminator architecture consists of several convolutional layers followed by batch normalization and activation functions (LeakyReLU for hidden layers and Sigmoid for the output layer). As the name suggests, it distinguishes between real and fake images.

## 4. Weight Initialization:
The `weights_init` function initializes the weights of convolutional and batch normalization layers. It is applied to both the generator and discriminator networks.

## 5. Loss Function:
Binary Cross Entropy Loss (BCELoss) is chosen as the loss function for both the generator and discriminator. It measures the binary cross entropy between the predicted output and the target labels.

## 6. Training Loops:
+ The training loop consists of alternating optimization between the discriminator and generator.
+ For each epoch and each batch, the discriminator is updated with real and fake data, followed by the generator being updated to fool the discriminator.
+ Losses are calculated and printed for monitoring training progress.

## 7. Saving models:
At the end of each epoch, the trained generator and discriminator models are saved. 
