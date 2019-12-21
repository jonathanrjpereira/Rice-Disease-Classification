![Banner](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Banner.svg)

It has been predicted that as global weather patterns begin to vary due to climate change, crop diseases are likely to become more severe and widespread. Hence, it is important to develop systems that quickly and easily analyze crops and identify a particular disease in order to limit further crop damage. Rice is one of the most popular staple food crops grown mainly across Asia, Africa and South America but is susceptible to a variety of pests and diseases. Physical characteristics such as decolorization of leaves can be used to identify several diseases that may affect the rice crop. For example, in the case of Brown-Spot, a fungal disease that affects the protective sheath of the leaves, the leaves are covered with several small oval brown spots with gray centers whereas, in the case of Leaf-Blast, the leaves are covered with larger brown lesions. Similarly, the leaves affected by the Rice Hispa pest can be identified by the long trail marks that develop on the surface of the leaf. The Convolutional Neural Network (CNN) is trained on a dataset consisting of images of leaves of both healthy and diseased rice plants. The images can be categorized into four different classes namely Brown-Spot, Rice Hispa, Leaf-Blast and Healthy. The dataset consists of 2092 different images with each class containing 523 images. Each image consists of a single healthy or diseased leaf placed against a white background.

![Diseases](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Diseases_2.svg)

## Features  
 - Classify diseased images of Rice leaves using Transfer Learning

## Background  
Prior methods for automatically classifying diseased plant images such as rule-based classifiers as used in [1], rely on a fixed set of rules to segment the leaf into affected and unaffected regions. Some of the rules to extract features involve observing the change in the mean and standard deviation between the color of the affected and unaffected regions. Rules to extract shape features involve individually placing several primitive shapes on top of the affected region and identifying the shape that covers the maximum area of the affected region. Once the features are extracted from the images, a set of fixed rules are used to classify the images depending upon the disease that may have affected the plant. The main drawback of such a classifier is that it will require several fixed rules for each disease which in turn could make it susceptible to noisy data.

The image classification technique described in this paper uses the basic structure of a CNN that consists of several convolutional layers, a pooling layer and a final fully connected layer. The convolutional layers act as a set of filters that extract the high-level features of the image. Max-pooling is one of the common methods used in pooling layers to reduce the spatial size of the extracted features thereby reducing the computation power required to calculate the weights for each layer. Finally, the extracted data is passed through a fully connected layer along with a softmax activation function which determines the class of the image.

## Proposed Method
We split the image dataset into training, validation and testing image sets. To prevent overfitting, we augment the training images by scaling and flipping the training images to increase the total number of training samples. In order to learn the features of the training images, we use a method called Transfer Learning wherein the ‘top’ layers of a pre-trained model are removed and replaced with layers that can learn the features that are specific to the training dataset. Transfer learning reduces the training time when compared to models that use randomly initialized weights. Our method uses six different pre-trained models namely, AlexNet, GoogLeNet, ResNet-50, Inception-v3, ShuffleNet and MobileNet-v2.    AlexNet consists of eight layers with weights, the first five being convolutional layers some of which are followed by max-pooling layers and the remaining three are fully-connected as shown in [2]. AlexNet produces a distribution over 1000 class labels and applies the ReLU function to the output of every convolutional and fully-connected layer. GoogLeNet is a 22-layer deep network consisting of several ‘Inception’ modules that are stacked upon each other with occasional max-pooling layers. The higher layers in the network usually Inception modules while the lower layers are usually convolutional layers as shown in [3]. ResNet-50 is based on VGGNet and contains several shortcut connections between convolutional layers forming a residual network as shown in [4]. The Inception-v3 network is 42 layers deep and has a lower computation cost when compared to VGGNet but has a computation cost that is approximately 2.5 greater than GoogLeNet when compared using the ILSVRC 2012 classification benchmark as shown in [5]. ShuffleNet is a CNN architecture built for mobile applications and consists of several stacked ‘Shuffle units’ that allows the network to incorporate group convolutions that obtain input data from different channel groups as shown in [6]. MobileNet-v2 is another CNN architecture built for mobile applications that uses residual connections in between bottleneck layers as shown in [7].

![Memory Size](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Memory%20Size.svg)

Depending upon the memory size required by each model, the pre-trained models are categorized into larger and smaller models. The smaller models consume less than 15MB and hence are better suited for mobile applications.


## Results  

![Training Time](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Training%20Time.svg)

Amongst the larger models, Inception-v3 had the longest training time of approximately 140 minutes whereas AlexNet had the shortest training time of approximately 18 minutes. Amongst the smaller mobile-oriented models, MobileNet-v2 had the longest training time of approximately 73 minutes whereas ShuffleNet had the shortest training time of approximately 38 minutes.

![Validation Accuracy](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Validation%20Accuracy.svg)

Amongst the larger models, Inception-v3 had the highest testing accuracy of approximately 72.1% whereas AlexNet had the lowest testing accuracy of approximately 48.5%. Amongst the smaller mobile-oriented models MobileNet-v2 had the highest testing accuracy of 62.5% whereas ShuffleNet had the lowest testing accuracy of 58.1%.

![Testing Accuracy](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Testing%20Accuracy.svg)

MobileNet-v2 performed significantly well when classifying images of Brown-Spot, Leaf-Blast and Healthy leaves while making several misclassifications for Rice Hispa with an accuracy of only 46.15%.

![MobileNet CM](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/MobileNet%20CM.svg)

Inception-v3 showed similar classification results as MobileNet-v2

![Inception CM](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Inception%20CM.svg)

The image below shows how the MobileNet-v2 model misclassifies an image of a grass leaf against a white background as Rice Hispa.

![Grass Test](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Grass.svg)

We also tested the accuracy of MobileNet-v2 on cropped images of Rice Hispa wherein the white background was minimized such that leaf occupies maximum area within the image. For cropped images of Rice Hispa, we observed an accuracy of approximately 80.81%.

![Hispa Cropped Test](https://github.com/jonathanrjpereira/Rice-Disease-Classification/blob/master/img/Hispa%20Cropped.svg)

## Conclusion
We described how we used transfer learning to classify images of diseased and healthy rice leaves. MobileNet-v2 had an accuracy of 62.5% and is best suited for mobile applications with memory and processing constraints. For cropped images of Rice Hispa, we observed a significant increase in the classification accuracy over uncropped test samples. Hence, we propose that real-world implementations of rice disease detection using convolutional neural networks must crop the test images to remove background noise in order to improve the 




## Examples
- [import.py](https://github.com/jonathanrjpereira) -



## Contributing
Are you a programmer, engineer or hobbyist who has a great idea for a new feature in this project? Maybe you have a good idea for a bug fix? Feel free to grab our code & schematics from Github and tinker with it. Don't forget to smash those ⭐️ & Pull Request buttons. [Contributor List](https://github.com/jonathanrjpereira/Rice-Disease-Classification/graphs/contributors)

Made with ❤️ by [Jonathan Pereira](https://github.com/jonathanrjpereira)

Banner Logo is Designed by [Freepik](https://www.freepik.com/) from [www.flaticon.com](www.flaticon.com)
Original Dataset by Huy Minh Do on [Kaggle](https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset).
