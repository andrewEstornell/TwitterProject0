# Project Overview
The following program is designed to perform sentiment analysis on tweets directed towards select companies. Namely, companies associated with the food service industry. This focus will expand with time.
Sentiment analysis is the process of deriving the underlying feeling or mood of given data. The sentiments extracted in this program are: good, bad, and neutral. A wider range of feelings can be included in sentiment analysis.
This analysis is performed via a neural network that has been trained on pre-labeled tweets that can be found in the file “TrainingData”. The neural network consists of convolutional layers that are densely connected. That is, each layer applies a filter to its given input and each layer takes input from all of it predecessors. Dense convolutional layers perform very well in analysing text. Another benefit of convolutional layers is the information gained from the kernel weights. These kernel weights, or filter weights, shed light on specific details the network finds important. One such example is a network trying to classify is a given image is a square or not. In the kernels of this network one would expect to find large positive weights representing horizontal and vertical lines, as these are defining features of squares. To a similar effect, one would expect to also have negative kernel weights representing curves or or oblong shapes. This same effect occurs when doing text analysis. We begin to see kernels that match letter combinations appearing in positively associated words.
-- Andrew
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
### Prerequisites
Python Package Index
```
$ pip install emoji
$ pip install python-twitter
```
### Installing
A step by step series of examples that tell you have to get a development env running
Say what the step will be
```
Give the example
```
And repeat
```
until finished
```
End with an example of getting some data out of the system or using it for a little demo
## Running the tests
Explain how to run the automated tests for this system
### Break down into end to end tests
Explain what these tests test and why
```
Give an example
```
### And coding style tests
Explain what these tests test and why
```
Example
```
## Deployment
Add additional notes about how to deploy this on a live system
## Built With
* [Python](https://www.python.org/downloads/release/python-364/) - 3.6.4
* [Twitter API](dev.twitter.com)
* [TensorFlow](https://www.tensorflow.org/)

## Authors
* **Andrew Estornell** - *Initial work* - [andrewestornell](https://github.com/andrewestornell)
* **Robert Plata** - *Initial work* - [robbieplata](http://github.com/robbieplata)
See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
