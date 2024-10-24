import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None  # Initialize the tree as None

    def author(self):
        return "ytsang6"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, train_x, train_y):
        """
        Add training data to learner
        """
        # Combine train_x and train_y into a single numpy array
        data = np.column_stack((train_x, train_y))
        # Build the decision tree
        self.tree = self.build_tree(data)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        results = []
        for point in points:
            node = 0  # Start at the root of the tree
            while not np.isnan(self.tree[node, 0]):
                i = int(self.tree[node, 0])
                split_val = self.tree[node, 1]
                if point[i] <= split_val:
                    node = int(node + self.tree[node, 2])
                else:
                    node = int(node + self.tree[node, 3])
            results.append(self.tree[node, 1])  # Add the leaf value to results
        return np.array(results)

    # def build_tree1(self, data):
    def build_tree(self, data):
        firstY = data[0, -1]  # Y as the last column to the right

        if data.shape[0] == 1:  # Create a leaf node if there is only one data point
            return np.array([[np.nan, firstY, np.nan, np.nan]])

        if np.all(data[:, -1] == data[0, -1]):  # Create a leaf node if all Y values are the same
            return np.array([[np.nan, firstY, np.nan, np.nan]])

        if data.shape[0] <= self.leaf_size:  # Create a leaf node if the number of data points is less than or equal to leaf_size
            return np.array([[np.nan, np.mean(data[:, -1]), np.nan, np.nan]])

        # Determine the random feature (i) based on the random choice
        random_feature_i = np.random.randint(data.shape[1] - 1)

        if [random_feature_i] is None:
            return np.array([[np.nan, np.mean(data[:, -1]), np.nan, np.nan]])

        split_val = np.median(data[:, random_feature_i])

        if data[:, random_feature_i].max() == data[:, random_feature_i].min():  # create leaf node if all random_feature_i are same [1,1,1,1,1], then there's no need to split
            return np.array([[np.nan, np.mean(data[:, -1]), np.nan, np.nan]])  # the Y for this leaf will be the mean of all Y

        left_data = data[data[:, random_feature_i] <= split_val]
        right_data = data[data[:, random_feature_i] > split_val]

        if left_data.shape[0] == data.shape[0] or right_data.shape[0] == data.shape[0]:
            return np.array([[np.nan, np.mean(data[:, -1]), np.nan, np.nan]])

        left_tree = self.build_tree(left_data) # left part
        right_tree = self.build_tree(right_data) # right part

        root = np.array([[random_feature_i, split_val, 1, left_tree.shape[0] + 1]])  # define root
        return np.concatenate((root, left_tree, right_tree), axis=0)  # return a tree
