# Hot Dog or Not Hot Dog
<div align="right"><i>Year: 2022</i></div>

<img src=https://user-images.githubusercontent.com/32619706/163104361-5be199bd-d997-4382-8018-3ba633f9f729.png>
Replicated the functionality of the <i>Not Hotdog</i> app (https://youtu.be/tWwCK95X6go) seen on Silicon Valley, a popular comedy series on HBO, by building a CNN to accurately classify images as <i>hot dog</i> or <i>not hot dog</i> through a pre-trained model: a case with limited training data.

## Data & Challenges
The dataset was obtained from [here](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog), which is a subset extracted from the *Food 101* dataset that contains photographs of 101 types of food. There are 498 images available for training, while the remaining 500 are to be kept aside for testing. For a classification task that is far from simple, there are extremely few samples to learn from. So, while this is a difficult machine learning problem, it is also a practical one: in many real-world use applications, even small-scale data collection can be prohibitively expensive or unattainable. Below is a sample from the dataset:

![Data Sample](https://user-images.githubusercontent.com/32619706/163095446-c892879d-8813-49e9-a89a-54cf29d6f44c.png)

## Data Pre-Processing & Loading
The data is directly loaded from the images on disk via Keras helper functions `ImageDataGenerator` and `flow_from_directory`. It performs two transformations:
* Rescaling pixels to be between [0, 1] or [-1, 1], depending on the model.
* Resizing images to be in `img_width`x`img_height` (150x150)

During training for each batch, the images are read from disk on the fly, loaded into memory and then the transformations are applied.

## Modeling
### Part (a): Simple CNN Model
To start with, a ConvNet model with a simple architecture is used to get an idea of the baseline performance upon training on limited data. It resulted in a mean test accuracy of 56.32% (+/- 0.82%) over 5 runs. 


**Change 1:** implementing data augmentation
 
<img src=https://user-images.githubusercontent.com/32619706/163098778-0f2614d9-ba42-48ab-a3a2-d077cd718456.png width="300">

In absence of sufficient training data, we can artificially augment the training data by applying random transformations to the originally available images. Though these are random, they are quite realistic, i.e., shearing the image by a small factor or flipping it horizontally. The idea is to prevent overfitting and helping the model generalize better. This can be interpreted based on the sample image below:

<img src=https://user-images.githubusercontent.com/32619706/163097492-c995cabe-f00d-464f-a739-330ab2452fc4.png>

Though an increase (~0.9%) in accuracy was observed, it wasn't significant.

**Change 2:** adding more hidden layers and setting a drop-out proportion of 20%

<img src=https://user-images.githubusercontent.com/32619706/163099266-8e892a5b-5b79-4562-a41d-ee10a68e2d9d.png width="250">

Minor increase in (~0.5%) in the accuracy; not satisfactory still.

Note: implementations of changes 1 and 2 are removed from the uploaded notebook for better readability.

### Part (b): Transfer Learning


It was realized that a more sophisticated method would be to use a network that has been pre-trained on a much larger dataset, since it would have already learned useful features, and thus allow achieving a higher accuracy than any method that relied just on the available data. This was done using a base-model ([Xception](https://arxiv.org/abs/1610.02357)) with pre-trained weights (on ImageNet). 

The workflow includes taking layers from the previously trained model and freezing them to avoid destroying the information contained by them during training, adding new trainable layers on top of the frozen ones, and training them on the new dataset, so they can turn the old features into predictions for this dataset.

<img src=https://user-images.githubusercontent.com/32619706/163101360-9997155f-9bde-4572-a8ab-a779a7f87834.png width="500">

Since Xception requires the input pixels to be scaled between -1 and 1, this step was also implemented through a keras rescaling layer which was set to non-training. Finally, a GlobalAveragePooling2D layer was added to reduce the trainable parameters without compromising on their effect.

## Results
<img src=https://user-images.githubusercontent.com/32619706/163102138-0979e40e-fd0b-486a-8436-1aaa58162b74.png width="500">

There was a **remarkable improvement** in the performance, with mean test accuracy reaching as high as **88.87%** (+/- 1.16%) over 5 runs, which is exceptionally good for the limited amount of training data at hand.
