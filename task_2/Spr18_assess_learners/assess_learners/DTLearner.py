import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from copy import deepcopy
from collections import Counter
from operator import itemgetter

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False, tree=None):
        # leaf size is used to define min. no of items must be in leaf node 
        self.leaf_size = leaf_size
        # verbose is used to show learner info.
        self.verbose = verbose
        self.tree = deepcopy(tree)
        if verbose:
            self.get_learner_info()

    def author(self):
        return 'Harshal'  # replace tb34 with your Georgia Tech username

    def get_learner_info(self):
        print ("Decision Tree Learner Information:")
        print ("Leaf size :", self.leaf_size)
        if self.tree is not None:
            print ("Tree shape:", self.tree.shape)
            print ("tree as a matrix:")
            # it creates dataframe from tree for human redable view
            dataframe_tree = pd.DataFrame(self.tree, columns=["factor", "split value", "left", "right"])
            dataframe_tree.index.name = "node"
            print (dataframe_tree)
        else:
            print ("No data in tree")


    def __build_tree(self, dataX, dataY, rootX=[], rootY=[]):
        # Get the number of samples (rows) and features (columns) of dataX
        no_of_samples = dataX.shape[0]
        no_of_features = dataX.shape[1]
        # If there is no sample left, return the most common value from the root of current node
        if no_of_samples == 0:
            return np.array([-1, Counter(rootY).most_common(1)[0][0], np.nan, np.nan])

        # If there are <= leaf_size samples or all data in dataY are the same, return leaf
        if no_of_samples <= self.leaf_size or len(pd.unique(dataY)) == 1:

            return np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])

        available_features_to_split = list(range(no_of_features))

        # Get a list of tuples of features and their correlations with dataY
        features_correlation = []
        for feature_i in range(no_of_features):
            abs_corr = abs(pearsonr(dataX[:, feature_i], dataY)[0])
            features_correlation.append((feature_i, abs_corr))

        # Sort the list in descending order by correlation
        features_correlation = sorted(features_correlation, key=itemgetter(1), reverse=True)

        # Choose the best feature, if any, by iterating over features_correlation
        feat_corr_i = 0
        while len(available_features_to_split) > 0:
            best_feature_i = features_correlation[feat_corr_i][0]
            best_abs_corr = features_correlation[feat_corr_i][1]

        scii   # Split the data according to the best feature
            split_val = np.median(dataX[:, best_feature_i])

            # left_index contains bool type array of dataX
            left_index = dataX[:, best_feature_i] <= split_val

            right_index = dataX[:, best_feature_i] > split_val

            # If we can split the data into two groups, then break out of the loop
            if len(np.unique(left_index)) != 1:
                break

            available_features_to_split.remove(best_feature_i)
            feat_corr_i += 1

        # If we complete the while loop and run out of features to split, return leaf
        if len(available_features_to_split) == 0:
            return np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])

        # Build left and right branches and the root
        lefttree = self.__build_tree(dataX[left_index], dataY[left_index], dataX, dataY)
        righttree = self.__build_tree(dataX[right_index], dataY[right_index], dataX, dataY)

        # Set the starting row for the right subtree of the current root
        if lefttree.ndim == 1:
            righttree_start = 2  # The right subtree starts 2 rows down
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        root = np.array([best_feature_i, split_val, 1, righttree_start])

        return np.vstack((root, lefttree, righttree))

    def __tree_search(self, point, row):

        # Get the feature on the row and its corresponding splitting value

        feat, split_val = self.tree[row, 0:2]

        # If splitting value of feature is -1, we have reached a leaf so return it
        if feat == -1:
            return split_val

        # If the corresponding feature's value from point <= split_val, go to the left tree
        elif point[int(feat)] <= split_val:
            pred = self.__tree_search(point, row + int(self.tree[row, 2]))

        # Otherwise, go to the right tree
        else:
            pred = self.__tree_search(point, row + int(self.tree[row, 3]))

        return pred

    def addEvidence(self, dataX, dataY):
        new_tree = self.__build_tree(dataX, dataY)

        # If self.tree is currently None, simply assign new_tree to it
        if self.tree is None:
            self.tree = new_tree

        # Otherwise, append new_tree to self.tree
        else:
            self.tree = np.vstack((self.tree, new_tree))

        # If there is only a single row, expand tree to a numpy ndarray for consistency
        if len(self.tree.shape) == 1:
            self.tree = np.expand_dims(self.tree, axis=0)

        if self.verbose:
            self.get_learner_info()

    def query(self, points):
        preds = []
        for point in points:
            preds.append(self.__tree_search(point, row=0))
        return np.asarray(preds)



if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
