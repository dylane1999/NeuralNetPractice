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
        '''
        train all nodes that need to be trained after going through the data
        :param train_data:
        :param train_labels:
        :return:
        '''
        # set initial weight
        if len(self.weights) != len(train_data[0]):
            self.reset_weights(len(train_data[0]))
        print(self.test_all(train_data, train_labels))
        while self.test_all(train_data, train_labels) != 1:
            print( self.test_all(train_data, train_labels))
            # iteratively update weights
            for i in range(len(train_data)):
                perceptron_output = self.test_example(train_data[i], train_labels[i])
                if self.is_update_required(perceptron_output, train_labels[i]):
                    # if update required, update node
                    self.weights = self.update_weight(perceptron_output, train_labels[i])  # update weights
        return


    def is_update_required(self, perceptron_output, label):
        if perceptron_output == label:
            return False
        return True

    def update_weight(self, perceptron_output, expected_output):
        new_weights = []
        for weight in self.weights:
            new_weights.append(weight + self.learning_rate*(expected_output - perceptron_output))
        return new_weights

    def get_opposite(self, value):
        if value == 1:
            return 0
        elif value == 0:
            return 1


    def test_example(self, data, label): # STUDENT SOLUTION
        '''
        takes a data point and a label, and returns 1 if the perceptron's output is above threshold, 0 otherwise.
        :param example:
        :return:  returns 1 if the perceptron's output is above threshold, 0 otherwise.
        '''
        weighted_sum = 0
        for i in range(len(data)):
            weighted_sum += data[i] * self.weights[i+1]  # get the weighted sum
        weighted_sum += self.bias * self.weights[0]  # add the bias
        if weighted_sum > self.threshold:
            return 1
        return 0

    def test_all(self, test_data, labels): # STUDENT SOLUTION
        '''
        test all data in test_data
        :param test_data:
        :param labels:
        :return: the overall accuracy (i.e., correct/total)
        '''
        total_outputs = len(test_data)
        correct_outputs = 0
        for i in range(len(test_data)):
            output = self.test_example(test_data[i], labels[i])
            if output == labels[i]:
                correct_outputs += 1
        accuracy = correct_outputs / total_outputs
        return accuracy



# pass

        # TODO


if __name__ == '__main__':
    threshold = 1
    learning_rate = .2
    bias = .4
    perceptron = Perceptron(threshold, learning_rate, bias)
    binary_data = [[0,0], [0,1], [1,1], [1,0]]
    and_labels = [0, 0, 1, 0]
    or_labels = [0, 1, 1, 1]
    nor_labels = [1, 0, 0, 0]
    nand_labels = [1, 1, 0, 1]

    x = perceptron.train_all(binary_data, and_labels)
    print(perceptron.weights)
    print(perceptron.test_all(binary_data, and_labels))


    # and_perceptron = Perceptron(threshold, learning_rate, bias)
    # and_perceptron.train_all(binary_data, and_labels)
    # print(and_perceptron.weights)
    # print(and_perceptron.test_all(binary_data, and_labels))
    #
    # or_perceptron = Perceptron(threshold, learning_rate, bias)
    # or_perceptron.train_all(binary_data, or_labels)
    # print(or_perceptron.weights)
    # print(or_perceptron.test_all(binary_data, or_labels))
    #
    # nor_perceptron = Perceptron(threshold, learning_rate, bias)
    # nor_perceptron.train_all(binary_data, nor_labels)
    # print(nor_perceptron.weights)
    # print(nor_perceptron.test_all(binary_data, nor_labels))
    #
    # and_perceptron = Perceptron(threshold, learning_rate, bias)
    # and_perceptron.train_all(binary_data, and_labels)
    # print(and_perceptron.weights)
    # print(and_perceptron.test_all(binary_data, and_labels))
