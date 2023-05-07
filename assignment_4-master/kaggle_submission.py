import numpy as np
from collections import Counter
import time

from numpy.random import sample


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, selected_feature, threshold, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label
        self.selected_feature = selected_feature
        self.threshold = threshold

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature, self.selected_feature, self.threshold):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None

    # TODO: finish this.
    decision_tree_2 = DecisionNode(None, None, lambda a: a[3] == 1)
    decision_tree_2.left = DecisionNode(None, None, None, 1)
    decision_tree_2.right = DecisionNode(None, None, None, 0)

    decision_tree_3 = DecisionNode(None, None, lambda a: a[3] == 1)
    decision_tree_3.left = DecisionNode(None, None, None, 0)
    decision_tree_3.right = DecisionNode(None, None, None, 1) 

    decision_tree_1 = DecisionNode(decision_tree_2, decision_tree_3, lambda a: a[2] == 1)
    decision_tree_root = DecisionNode(None, decision_tree_1, lambda a: a[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    vector = np.array(classifier_output) + np.array(true_labels)
    counts = Counter(vector)
    true_positive = counts[2]
    true_negative = counts[0]

    vector = np.array(classifier_output) - np.array(true_labels)
    counts = Counter(vector)
    false_positive = counts[1]
    false_negative = counts[-1]
    c_matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])

    return c_matrix

    # raise NotImplemented()


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    c_matrix = confusion_matrix(classifier_output, true_labels)
    p = c_matrix[0, 0] / (c_matrix[0, 0] + c_matrix[1, 0])

    return p

    # raise NotImplemented()


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    c_matrix = confusion_matrix(classifier_output, true_labels)
    r = c_matrix[0, 0] / (c_matrix[0, 0] + c_matrix[0, 1])

    return r
    # raise NotImplemented()


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    c_matrix = confusion_matrix(classifier_output, true_labels)
    a = (c_matrix[0, 0] + c_matrix[1, 1]) / len(classifier_output)

    return a
    # raise NotImplemented()


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    counts  = Counter(class_vector)
    prob = np.array([(v/sum(counts.values()))**2 for v in counts.values()])
    gini_i = 1 - sum(prob)

    return gini_i

    # raise NotImplemented()


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    gini_c = 0
    n = 0
    for i in current_classes:
        n += len(i)

    for i in current_classes:
        gini_c += gini_impurity(i) * len(i)/n

    gini_p = gini_impurity(previous_classes)

    return gini_p - gini_c
    # raise NotImplemented()


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        previous_class = list(classes)
        m, n = np.atleast_2d(features).shape
        if np.sum(classes) == 0 or np.sum(classes) == m:
            # return DecisionNode(None, None, None, classes[0])
            return DecisionNode(None, None, None, None, None, classes[0])

        elif depth == self.depth_limit:
            counts = Counter(classes)

            if counts[1]/m >= 0.5:
                # return DecisionNode(None, None, None, 1)
                return DecisionNode(None, None, None, None, None, 1)
            else:
                # return DecisionNode(None, None, None, 0)
                return DecisionNode(None, None, None, None, None, 0)

        else:
            threshold = 0
            data = np.hstack((features, classes.reshape((m,1))))
            alpha_max = -np.inf
            for i in range(n):
                sorted_data = data[np.lexsort([data.T[i]])]
                sorted_attr = np.sort(np.array(list(set(sorted_data[:, i]))))
                k = sorted_attr.shape[0]
                alpha = -np.inf
                
                for j in range(1, k):
                    arg = list(sorted_data[:, i]).index(sorted_attr[j])
                    sub_class1 = list(sorted_data[:arg, -1])
                    sub_class2 = list(sorted_data[arg:, -1])
                    gain = gini_gain(previous_class, [sub_class1, sub_class2])
                    iv = -(len(sub_class1)/m * np.log2(len(sub_class1)/m) + len(sub_class2)/m * np.log2(len(sub_class2)/m))
                    g = gain/iv

                    if g > alpha:
                        alpha = g
                        t = 0.5 * (sorted_data[arg-1, i] + sorted_data[arg, i])
                        sc1 = np.array(sub_class1)
                        sc2 = np.array(sub_class2)
                        sub_features1 = sorted_data[:arg, :-1]
                        sub_features2 = sorted_data[arg:, :-1]

                if alpha > alpha_max:
                    alpha_max = alpha
                    threshold = t
                    class_left = sc1
                    class_right = sc2
                    features_left = sub_features1
                    features_right = sub_features2
                    selected_feature = i

            if alpha_max == -np.inf:
                counts = Counter(classes)
                if counts[1]/m >= 0.5:
                    # return DecisionNode(None, None, None, 1)
                    return DecisionNode(None, None, None, None, None, 1)
                else:
                    # return DecisionNode(None, None, None, 0)
                    return DecisionNode(None, None, None, None, None, 0)
                    
            decision_tree_node_left = self.__build_tree__(features_left, class_left, depth + 1)
            decision_tree_node_right = self.__build_tree__(features_right, class_right, depth + 1)
            # decision_tree_node = DecisionNode(decision_tree_node_left, decision_tree_node_right, lambda a: a[selected_feature] < threshold)
            decision_tree_node = DecisionNode(decision_tree_node_left, decision_tree_node_right, self.func, selected_feature, threshold)
            
            return decision_tree_node
        # raise NotImplemented()

    def func(self, feature, selected_feature, threshold):
        return feature[selected_feature] < threshold

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = []

        # TODO: finish this.
        # raise NotImplemented()
        m = features.shape[0]
        for index in range(m):
            decision = self.root.decide(features[index])
            class_labels.append(decision)

        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    m = dataset[0].shape[0]
    data = np.hstack((dataset[0], dataset[1].reshape((m,1))))
    r = m%k
    n = m//k
    ls = np.arange(m)
    np.random.shuffle(ls)
    sub = 0
    folds = []
    
    for i in range(k):
        test = data[ls[sub:(sub + n)], :]
        test_set = (test[:, :-1], test[:, -1])
        if sub == 0:
            training = data[ls[(sub + n):], :]
        else:
            training = np.vstack((data[ls[0:sub], :], data[ls[(sub + n):], :]))
        
        training_set = (training[:, :-1], training[:, -1])
        sub = sub + n

        folds.append((training_set, test_set))
    
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        m, n = np.atleast_2d(features).shape
        data = np.hstack((features, classes.reshape((m,1))))
        tree_samples_num = int(np.round(self.example_subsample_rate * m))
        attr_samples_num = int(np.round(self.attr_subsample_rate * n))

        self.attr_used = []
        # self.attr_used = [(0, 1), (2, 0), (2, 1), (1, 3), (0, 1)]

        for i in range(self.num_trees):
            nums_tree = np.random.randint(m, size=tree_samples_num)
            samples = data[nums_tree,:]

            nums_attr = np.arange(n)
            np.random.shuffle(nums_attr)
            nums_attr = nums_attr[:attr_samples_num]

            self.attr_used.append(tuple(nums_attr))

            # nums_attr = self.attr_used[i]

            tree = DecisionTree(self.depth_limit)
            tree.fit(samples[:, nums_attr], samples[:, -1])
            self.trees.append(tree)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        m, n = np.atleast_2d(features).shape
        output = np.zeros(m)
        for i in range(self.num_trees):
            current_tree = self.trees[i]
            output += np.array(current_tree.classify(features[:, self.attr_used[i]]))
        
        output = output/self.num_trees - 0.5
        answers = list(np.int64(output>0))

        return answers



class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        # raise NotImplemented()
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample = attr_subsample

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        # raise NotImplemented()
        # self.tree = DecisionTree()
        # self.tree.fit(features, classes)

        m, n = np.atleast_2d(features).shape
        data = np.hstack((features, classes.reshape((m,1))))
        tree_samples_num = int(np.round(self.example_subsample_rate * m))
        attr_samples_num = self.attr_subsample

        self.attr_used = []

        for i in range(self.num_trees):
            nums_tree = np.random.randint(m, size=tree_samples_num)
            samples = data[nums_tree,:]

            nums_attr = np.arange(n)
            np.random.shuffle(nums_attr)
            nums_attr = nums_attr[:attr_samples_num]

            self.attr_used.append(tuple(nums_attr))

            tree = DecisionTree(self.depth_limit)
            tree.fit(samples[:, nums_attr], samples[:, -1])
            self.trees.append(tree)
        # return tree


    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        # raise NotImplemented()
        # output = self.tree.classify(features)
        # return output

        m, n = np.atleast_2d(features).shape
        output = np.zeros(m)
        for i in range(self.num_trees):
            current_tree = self.trees[i]
            output += np.array(current_tree.classify(features[:, self.attr_used[i]]))
        
        output = output/self.num_trees - 0.5
        answers = list(np.int64(output>0))

        return answers


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        vectorized = data * data + data

        return vectorized
        # raise NotImplemented()


    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        sum_array = np.sum(data[:100, :], axis=1)
        max_sum = np.max(sum_array)
        max_sum_index = np.argmax(sum_array)

        return max_sum, max_sum_index
        # raise NotImplemented()

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        flattened = data.flatten()
        positive_num = filter(lambda x: x > 0, flattened)
        unique_dict = Counter(positive_num)

        return unique_dict.items()
        # raise NotImplemented()

def return_your_name():
    # return your name
    # TODO: finish this
    return "Wenda Xu"
    # raise NotImplemented()
