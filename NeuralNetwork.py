import tensorflow as tf
import numpy as np


class NeuralNetwork:

    def __init__(self, fileName):
        self.output = None  # List of classifications: good, neutral, or bad
        self.input = None  # List of Tweet objects that were fed into the network
        self.error = 0  # Incorrectness of the networks guesses, i.e. error = ||Outputs - TrueLables||^2
        self.numberOfClasses = 3  # Number of potential outputs, i.e. |{good=1, neutral=0, bad=-1}|
        self.tweetsPerBatch = 150
        self.data, self.labels = self.createData(fileName)
        self.iterations = 100


    def createNetwork(self):
        """
        Create the neuralnetwork and its layers
        :param tweetLength: length of a tweet
        :return: None
        """

        # Sizes that will be used for the convolutional and fully connected layers.
        numberOfConvolutionalLayers = 5
        filterLengths = [2, 3, 4, 5, 6]
        filterHeights = [6, 6, 6, 6, 6]
        numberOfFilters = [10, 10, 10, 10, 10]
        numberOfFullyConnectedLayers = 2
        fullyConnectedSize = 180
        # Characters of each tweet are converted to binary arrays, the largest binary number needed has order 5
        tweetHight = 5
        # Tweets can be a maximum of 240 characters
        tweetLength = 240
        inputSize = tweetHight * tweetLength

        # Arbitrary number of vectors with length tweetLength
        x = tf.placeholder(tf.float32, [None, inputSize], name='x')
        # Reshapes x to be 3d so that convolution can be applied
        xTweet = tf.reshape(x, [-1, tweetLength, tweetHight, 1])
        # Tensor storing all the true labels for each tweet in our training batch
        yTrue = tf.placeholder(tf.float32, shape=[None, self.numberOfClasses], name='yTrue')

        yTrueClass = tf.argmax(yTrue, dimesnion=1)

        # Constructs the computational graph.
        # This graph will be set up such that Layers = {1, 2, 3, ..., n}, where 1 is the first layer, and n is the last
        # The set of directed connections will be Connections = {(a, b): a,b\in Layers && a < b }
        # That is each layer is connected to every layer in front of it
        # We do this for two reasons, the first being that is solves the problem of the vanishing gradient
        #                             the second being that it helps give more equal priority to all convolutions
        convolutionalLayers = [numberOfConvolutionalLayers]
        convolutionalWeights = [numberOfConvolutionalLayers]
        # Starting input, input will be updated such that each layer takes in input from all previous layers
        input = xTweet
        # Similarly the number of input channels will be updated
        numberOfInputChannels = 1

        # Creates the convolutional layers
        for i in range(numberOfConvolutionalLayers):
            convolutionalLayers[i], convolutionalWeights[i] = \
                self.newConvolutionalLayer(input=input, numberOfInputChannels=numberOfInputChannels,
                                           filterHight=filterHeights[i], filterLength=filterLengths[i],
                                           numberOfFilters=numberOfFilters[i], use2x2pooling=False)

            # Adds the output from the current layer to the tensor of all the previous layers' outputs
            input = tf.concat(input, convolutionalLayers[i])

            # Updates the number of input channels so that all previous layers' inputs are taken into account
            numberOfInputChannels = 0
            for j in range(i):
                numberOfInputChannels += numberOfFilters[j]

        fullyConnectedLayers = [numberOfFullyConnectedLayers]
        numberOfFeautres = [numberOfFullyConnectedLayers]

        # Reshapes the input tensor to 1D so that it can be run through a fully connected layer
        flattenedInput = self.flattenLayer(input)

        # Creates a fully connected layer
        fullyConnectedLayers[0] = self.newFullyConnectedLayer(input=flattenedInput, numberOfInputs=numberOfFeautres,
                                                              numberOfOutputs=fullyConnectedSize, useRelu=True)

        # Updates the input and number of inputs
        flattenedInput = tf.concat(flattenedInput, fullyConnectedLayers[0])
        numberOfFeautres += fullyConnectedSize

        # Creates the last layer of the network
        fullyConnectedLayers[1] = self.newFullyConnectedLayer(input=flattenedInput, numberOfInputs=numberOfFeautres,
                                                              numberOfOutputs=self.numberOfClasses, useRelu=True)
        return fullyConnectedLayers[1], x, yTrue, yTrueClass

    def runNetwork(self):
        network, x, yTrue, yTrueClass = self.createNetwork()
        yPrediction = tf.nn.softmax(network)

        # Gives the predicted class, that is [1, 0, 0] , [0, 1, 0], [0, 0, 1]
        yPreditionClass = tf.argmax(yPrediction, dimension=1)

        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=yTrue)

        cost = tf.reduce_mean(crossEntropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

        correctPrediction = tf.equal(yPreditionClass, yTrueClass)

        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        session = tf.Session()

        session.run(tf.initialize_all_variables())

        saver = tf.train.Saver()

        self.optimization(numberOfIterations=100, x=x, y=yTrue, session=session, optimizer=optimizer, accuracy=accuracy)
        self.displayAccuracy('testData.text', x=x, y=yTrue, session=session, yPredictionClass=yPreditionClass)

    def newWeights(self, shape):
        """
        Generates new weights for a layer
        :param shape: dimensions of the weight tensor
        :return: a tensor of weights
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.5))

    def newBiases(self, shape):
        """
        Generates new biases for a layer
        :param shape: dimensions of the biases tensor
        :return: a tensors of biases
        """
        return tf.Variable(tf.constant(0.05, shape=[shape]))

    def newConvolutionalLayer(self, input, numberOfInputChannels, filterHeight, filterLength, numberOfFilters,
                              use2x2pooling=True):
        """
        Creates a new convolutional layer
        :param input: previous layer's output
        :param numberOfInputChannels:  number of outputs (number of neurons) in the previous layer
        :param numberOfFilters: number of such filters
        :param use2x2pooling: downscales the output tensor such that each quadrant of 4 is is squished down to a single unit
        :return: a tensor that is the output of the layer along with the associated weights
        """
        # Shape out the filters' weights
        weightShape = [filterHeight, filterLength, numberOfInputChannels, numberOfFilters]

        # Shape of the filters biases
        biasesShape = [numberOfFilters]

        # Creates containers for weights and biases of the filters
        weights = self.newWeights(shape=weightShape)
        biases = self.newBiases(shape=biasesShape)

        # Initializes a 2d convolutional layer
        # strides = [1, y-axis movement units, x-axis movement units, 1]
        # padding = 'SAME' => tensor is given extra dimensions of zeros if needed
        layer = tf.nn.conv2d(inptut=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Adds biases to layer
        layer += biases

        if use2x2pooling:
            # Adds a pooling filter to the layer
            # pooling takes each 2x2 region, say R, and projects it to a 1x1 region, say B, via B = max(R)
            layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Adds a ReLU activation function to the layer, ReLU = Rectified Linear Unit
        # ReLU activation changes the original output, say V, to O = max(0, V), i.e. all negative outputs are set to 0
        layer = tf.nn.relu(layer)

        return layer, weights

    def flattenLayer(self, layer):
        """
        The output of convolutional layers is tensors with 4 or dimensions
        The function will scale down the dimension of the output tensor so that it may be feed into the next layer
        :param layer: current layer's output
        :return: flattened output and number of features of the current layer
        """

        # Get dimension of the layer
        layerShape = layer.get_shape()

        numberOfFeatures = np.array(layerShape[1:4], dtype=int).prod()

        # Reshapes the current layer's output
        flatLayer = tf.reshape(layer, [-1, numberOfFeatures])
        return flatLayer, numberOfFeatures

    def newFullyConnectedLayer(self, input, numberOfInputs, numberOfOutputs, useRelu=True):
        """
        Creates a new fully connected layer
        :param input: previous layer of the model
        :param numberOfInputs: number of outputs(neurons) from the previous layer
        :param numberOfOutputs: number of outputs of this layer
        :param useRelu: if true then output = max(0, output)
        :return: a new fully connected layer
        """

        # Initializes new containers for weights and biases
        weights = self.newWeights(shape=[numberOfInputs, numberOfOutputs])
        biases = self.newBiases(shape=[numberOfOutputs])

        # Creates a new layer which is the matrix multiplication of the weights summed with the biases matrix
        layer = tf.matmul(input, weights) + biases

        # Adds ReLU activation, that is output = max(output, 0)
        if useRelu:
            layer = tf.nn.relu(layer)

        return layer

    def createData(self, fileName):
        # List of tweets and their associated labels
        tweets = []
        labels = []
        file = open(fileName)
        for line in file:
            # Tweet stored as a 2d binary array
            binaryTweet = []
            # True label of the tweet, i.e. label \in {good, bad, neutral}
            label = []
            # Parses each of the three possible line types in file
            if line.startswith('*********'):
                # A row of *'s is used as dividers between tweets
                continue
            if line.startswith('---'):
                # Assigns each tweet a one-hot vector label
                if line == '---G---':
                    label = [1.0, 0.0, 0.0]
                elif line == '---N---':
                    label = [0.0, 1.0, 0.0]
                elif line == '---B---':
                    label = [0.0, 0.0, 1.0]
            else:
                # Converts the actual tweet into a 2d binary array
                binaryTweet = self.processTweet(line)

            # Stores the new converted tweet and associated label in the dictionary
            tweets.append(binaryTweet)
            labels.append(label)
        return tweets, labels

    def nextBatch(self):
        """
        Selects a given number of random samples from our data and their associated labels
        :return: list of randomly selected data and a list of randomly selected labels
        """
        index = np.arange(0, len(self.data))
        np.random.shuffle(index)
        index = index[:self.tweetsPerBatch]
        shuffledData = [self.data[ i] for i in index]
        shuffledLabels = [self.labels[ i] for i in index]
        return np.asarray(shuffledData), np.asarray(shuffledLabels)

    def displayAccuracy(self, testFileName, x, y, session, yPredictionClass):
        numberOfTests = 150
        classPredictions = np.zeros(shape=numberOfTests, dtype=np.int)
        file = open(testFileName)

        tweets, labels = self.createData(testFileName)
        feedDict = {x: tweets, y: labels}
        classPredictions = session.run(yPredictionClass, feed_dict=feedDict)

        correct = (labels == classPredictions)
        correctSum = correct.sum()
        print(float(correctSum) / numberOfTests)

    def optimization(self, numberOfIterations, x, y, session, optimizer, accuracy):
        """
        Trains the network by repeatedly selecting random batches of training data, calculating the forward pass of these
        batches, then back propagating the error through the network.
        :param numberOfIterations: how many batches the training will consist of
        :param x: tf placeholder for the input
        :param y: tf placeholder for the lables
        :param session: tf session
        :param optimizer: determines how to optimize the network
        :param accuracy: measures accuracy of the network
        :return: Nothing
        """
        global currentIteration
        for i in range(currentIteration, currentIteration + numberOfIterations):
            batch, labels = self.nextBatch()
            # Creates a dictionary of our data and its associated labels
            feedDict = {x: batch, y: labels}

            # Runs a tensorflow session on the selected batch of data
            session.run(optimizer, feed_dict=feedDict)

            # Displays progress
            if i % 50 == 0:
                # Calculates and displays the accuracy of the network
                currentAccuracy = session.run(accuracy, feed_dict=feedDict)
                print("Current Iteration " + str(i) + " Accuracy = " + str(currentAccuracy))

        currentIteration += numberOfIterations

    def alex_net(self):
        """
            Creates an alexnet model
        :return: directed graph of network.
        """
        tweet_length = 140
        tweet_height = 5
        input_size = 8 * 140  # make binary power * max characters

        # Sizes for the layers
        filters_0 = 20
        filters_1_1 = 10
        filters_1_2 = 10
        filters_2_1 = 20
        filters_2_2 = 20
        filters_3_1 = 10
        filters_3_2 = 10



        # Arbitrary number of vectors with length tweetLength
        x = tf.placeholder(tf.float32, [None, input_size], name='x')
        # Reshapes x to be 3d so that convolution can be applied
        x_tweet = tf.reshape(x, [-1, tweet_length, tweet_height, 1])
        # Tensor storing all the true labels for each tweet in our training batch
        y_true = tf.placeholder(tf.float32, shape=[None, self.numberOfClasses], name='yTrue')

        yTrueClass = tf.argmax(y_true, dimesnion=1)

        # Similarly the number of input channels will be updated
        numberOfInputChannels = 1

        # Creates the convolutional layers
        for i in range(numberOfConvolutionalLayers):
            convolutionalLayers[i], convolutionalWeights[i] = \
                self.newConvolutionalLayer(input=input, numberOfInputChannels=numberOfInputChannels,
                                           filterHight=filterHeights[i], filterLength=filterLengths[i],
                                           numberOfFilters=numberOfFilters[i], use2x2pooling=False)

            # Adds the output from the current layer to the tensor of all the previous layers' outputs
            input = tf.concat(input, convolutionalLayers[i])

            # Updates the number of input channels so that all previous layers' inputs are taken into account
            numberOfInputChannels = 0
            for j in range(i):
                numberOfInputChannels += numberOfFilters[j]

        fullyConnectedLayers = [numberOfFullyConnectedLayers]
        numberOfFeautres = [numberOfFullyConnectedLayers]

        # Reshapes the input tensor to 1D so that it can be run through a fully connected layer
        flattenedInput = self.flattenLayer(input)

        # Creates a fully connected layer
        fullyConnectedLayers[0] = self.newFullyConnectedLayer(input=flattenedInput, numberOfInputs=numberOfFeautres,
                                                              numberOfOutputs=fullyConnectedSize, useRelu=True)

        # Updates the input and number of inputs
        flattenedInput = tf.concat(flattenedInput, fullyConnectedLayers[0])
        numberOfFeautres += fullyConnectedSize

        # Creates the last layer of the network
        fullyConnectedLayers[1] = self.newFullyConnectedLayer(input=flattenedInput, numberOfInputs=numberOfFeautres,
                                                              numberOfOutputs=self.numberOfClasses, useRelu=True)