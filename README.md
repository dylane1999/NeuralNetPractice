# P4: Linear Classifiers and Deep Learning
Collaborator(s): 

In this project, we will use `pytorch`, a popular python library for deep learning. Installation instructions can be found [here](https://pytorch.org/get-started/locally/). `pytorch` requires a 64-bit python 3.x. 
Note: unless you have an NVIDIA graphics card, you should _not_ install a CUDA version (choose CPU instead).
You will also want to install `matplotlib`, which is useful for visualization. Installation instructions can be found [here](https://matplotlib.org/stable/users/installing.html)
As always, it is up to you whether you want to use Anaconda, another package manager, or no package manager.

For any answers that don't involve editing a .py file, please edit this README directly (on your branch). For anything that I ask you to print, please paste the output in this file or attach a screen shot.
    
1. For the first part of this assignment, we will write a Perceptron class. To keep things simple, we will use a threshold unit.    
a. Take a look at the starter code in Perceptron.py. What are the member variables? What does each do?  

- self.threshold = threshold
a perceptron takes in inputs, applies a linear combination, if the combination is greater or lesser than some threshold value, produces an ouput of 1 or 0 respectively. The Threshold is the number required to create the response of a 1

- Self.learning_rate
The learning rate is the parameter that alters how fast the model changes weights relative to the amount of training epochs. A learning rate that is too large can cause the model to converge too quickly to a suboptimal solution, whereas a learning rate that is too small can cause the process to get stuck.

- Self.bias
The bias value allows the activation function to be shifted to the left or right, to better fit the data. changes to the weights alter the steepness of the sigmoid curve, whilst the bias offsets it, shifting the entire curve so it fits better.

- Self.default_weight
Weight is the parameter within a neural network that transforms input data within the network's hidden layers. Inputs get multiplied by a weight value and the resulting output is either observed, or passed to the next layer in the neural network. This is the default weight for the nodes that is a random weight 0-1

- Self.weights
An array of n weights for n number of nodes in the neural network.  

b. What does the `reset_weights()` method do? (Note: this method uses python's list comprehension. Feel free to unpack it.)  
- The reset weights method fills the self.weights array with n number of random weights for a given size n 

c. Perhaps counterintuitively, we will start by writing the `test_example()` method. `test_example()` takes a data point and a label, and returns 1 if the perceptron's output is obove threshold, 0 otherwise.

   What is the equation for figuring out whether a data point is above threshold?  
   - Output = 0 if weight*data point + bias < threshold
   - Output = 1 if weight*data point + bias >= threshold

   d. Fill in the code for `test_example()`
   - complete
   
   e. To wrap up the testing code, fill in the `test_all()` method, which takes a list of data points and a corresponding list of labels.
        Yes, this is just a loop that calls `test_example()`. It should return the overall accuracy (i.e., correct/total)
   - complete
        
f. We will first fill in the `train_all()` method, which takes a list of data points and a list of labels corresponding to the datapoints.
   What does the code that's filled out for you do? Why do we need it?  
   - this code checks the length of weights and if it is not equal to the size of row 0 of the training data, then it fills it up with the random weights for size of row 0 of the training data. 
   
   g. For our `train_all()` method, we will update weights iteratively. In other words, we will update weights after each training example that requires an update.
   Given a training example, how do you know whether to update weights?  
   
   h. What is the equation for updating weights, given a training example?  
   
   i. For every training example passed into `train_all()`, we will need to check whether the weights need to be updated, then update them according to the equation above.
   Write the code to do so. You may reuse existing methods and add any helper methods you wish.  
   
   j. We will use an accuracy of 1 to test for convergence. Write a loop around the above code that updates weights until convergence.
   
k. The `__main__` method in Perceptron.py includes data sets for logical "and", "or", "nand", and "nor". Test your perceptron on them (you may choose all input parameters). Does it always converge?
   (Hint: don't forget to reset weights if not instantiating a new perceptron for each)  
   
l. Why is using an accuracy of 1 a bad test of convergence in a general sense? What are some better alternatives?
   
2. Documentation for pytorch can be found [here](https://pytorch.org/docs/stable/index.html). There are also many tutorial available [here](https://pytorch.org/tutorials/). We will focus on the [basics tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html). Use PyTorch_Playground.py.    
a. The first section of the tutorial explains how data are represented in `pytorch`. Why is it important to start with data representation?  
   b. Follow the instructions to download the FashionMNIST train and test data, and create DataLoader objects. Why do we need the DataLoader class? 
   c. Next, move on to the "Creating Models" section of the tutorial. How many layers are in the neural network defined here? (Reminder: We touched upon the ReLU function very briefly in class.Think of it as an alternative to the sigmoid)  
   d. Copy the code for the NeuralNetwork class into your PyTorch_Playground.py file. Run your code. Make sure you understand what it's doing.  
   e. Now, it's time to define the training method. This is described in the "Optimizing the Model Parameters" section of the tutorial. Intuitively, why is optimization and training the same thing?  
   f. How well does your model do on the FashionMNIST test set? If you increase the number of training epochs to 10, does it improve?    
   f. Follow the "Save Models" instructions to save your model to a file called "model.pth". Include it in your submission.  
   g. Pick a different [vision dataset](https://pytorch.org/vision/stable/index.html). Create only a _test_ set from this dataset, and test your model. How well does it perform?  
   h. Now create a training set from your new dataset. Train and test and new model (don't worry about changing the network architecture). Are the results different? Save your new model to "model2.pth" and include in your submission.
