import random

class Perceptron:

    def __init__(self, threshold, learning_rate, bias, default_weight=None):
        # assumes labels for all data are [0 if <thresh,1 if >= thresh]
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.bias = bias
        self.default_weight = default_weight  # default None --> random weight (0,1)
        self.weights = []

    def reset_weights(self, size):
        self.weights = [self.default_weight if self.default_weight is not None else random.random() for i in (range(size+1))]

    def train_all(self, train_data, train_labels): # STUDENT SOLUTION
        # set initial weight
        if len(self.weights) != len(train_data[0]):
            self.reset_weights(len(train_data[0]))

        # TODO


    def test_example(self, example): # STUDENT SOLUTION
        pass

        # TODO

    def test_all(self, test_data, labels): # STUDENT SOLUTION
        pass

        # TODO


if __name__ == '__main__':
    binary_data = [[0,0], [0,1], [1,1], [1,0]]
    and_labels = [0, 0, 1, 0]
    or_labels = [0, 1, 1, 1]
    nor_labels = [1, 0, 0, 0]
    nand_labels = [1, 1, 0, 1]

    
