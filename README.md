# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output. An autoencoder will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.The goal of an autoencoder is to get an output that is identical to the input. Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image. We are using MNIST Dataset for this experiment. The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![276110167-bcf7dd70-fb7b-41d9-9010-1aa95afedc2a](https://github.com/Bmohamedathil/convolutional-denoising-autoencoder/assets/119560261/528a0a98-1786-4498-8e23-2d0788c9c132)

## Convolution Autoencoder Network Model
![276110230-6a07a66a-4bb4-4e8d-8a34-101bf313d37b](https://github.com/Bmohamedathil/convolutional-denoising-autoencoder/assets/119560261/51c574a0-4e0f-439c-bf0f-6182f8cf4eb3)


## DESIGN STEPS

### STEP 1:
Import the necessary libraries and dataset.
### STEP 2:
Load the dataset and scale the values for easier computation.
### STEP 3:
Add noise to the images randomly for both the train and test sets.
### STEP 4:
Build the Neural Model using Convolutional Layer,Pooling Layer,Up Sampling Layer. Make sure the input shape and output shape of the model are identical.
### STEP 5:
Pass test data for validating manually.
### STEP 6:
Plot the predictions for visualization.
## PROGRAM
```
Developed By: MOHAMED ATHIL B
Register No : 212222230081
```
### Import the packages :
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```
### Read Dataset & Scale it :
```
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
```
### Add Noise to Image :
```
noise_factor = 0.5
x_train_noisy =x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy =x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```
### Plot the Images :
```
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
### Develop a Autoencoder DL Model :
```
input_img = keras.Input(shape=(28, 28, 1))
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2, 2), padding='same')(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()
```
### Compile & Fit the model :
```
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```
### Plot Metrics Graphs :
```
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()
```
### Predict Using the Model :
```
decoded_imgs = autoencoder.predict(x_test_noisy)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![276111634-b0e57ae7-c2d3-4a8e-97ab-78d188dfbfcf](https://github.com/Bmohamedathil/convolutional-denoising-autoencoder/assets/119560261/28b5be4e-4a4d-4124-be53-3ee254aafac2)
### Model Summary :
![276111749-17c23457-59b3-40e5-a2f9-d12a58552de3](https://github.com/Bmohamedathil/convolutional-denoising-autoencoder/assets/119560261/c5d3375c-3d4c-4bc4-872a-b10b40650837)

### Original vs Noisy Vs Reconstructed Image
![276112089-ddf10439-c150-46ac-804e-7af7c0284e54](https://github.com/Bmohamedathil/convolutional-denoising-autoencoder/assets/119560261/a9dfbda2-690a-41af-8639-a52edb5acc64)

## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
