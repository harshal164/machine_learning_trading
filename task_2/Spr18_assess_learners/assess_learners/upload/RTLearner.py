import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False, tree=None):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = deepcopy(tree)
        if verbose:
            self.get_learner_info()
    def author(self):
        return 'Harshal'  # replace tb34 with your Georgia Tech username

    def __build_tree(self, dataX, dataY):
        number_of_samples = dataX.shape[0]
        num_feats = dataX.shape[1]
        # Leaf value is the most common dataY
        leaf = np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])
        # If there are <= leaf_size samples or all data in dataY are the same, return leaf
        if number_of_samples <= self.leaf_size or len(np.unique(dataY)) == 1:
            return leaf
        available_features_to_split = list(range(num_feats))

        while len(available_features_to_split) > 0:
            # Randomly choose a feature to split on
            rand_feat_i = np.random.choice(available_features_to_split)

            # Randomly choose two rows
            rand_rows = [np.random.randint(0, number_of_samples), np.random.randint(0, number_of_samples)]

            # If the two rows are the same, reselect them until they are different
            while rand_rows[0] == rand_rows[1] and number_of_samples > 1:
                rand_rows = [np.random.randint(0, number_of_samples), np.random.randint(0, number_of_samples)]

            # Split the data by computing the mean of feature values of two random rows
            split_val = np.mean([dataX[rand_rows[0], rand_feat_i],
                                 dataX[rand_rows[1], rand_feat_i]])

            # Logical arrays for indexing
            left_index = dataX[:, rand_feat_i] <= split_val
            right_index = dataX[:, rand_feat_i] > split_val

            # If we can split the data into two groups, then break out of the loop
            if len(np.unique(left_index)) != 1:
                break

            available_features_to_split.remove(rand_feat_i)

        # If we complete the while loop and run out of features to split, return leaf
        if len(available_features_to_split) == 0:
            return leaf

        # Build left and right branches and the root
        lefttree = self.__build_tree(dataX[left_index], dataY[left_index])
        righttree = self.__build_tree(dataX[right_index], dataY[right_index])

        # Set the starting row for the right subtree of the current root
        if lefttree.ndim == 1:
            righttree_start = 2  # The right subtree starts 2 rows down
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        root = np.array([rand_feat_i, split_val, 1, righttree_start])
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

    def get_learner_info(self):
        print ("leaf_size =", self.leaf_size)
        if self.tree is not None:
            print ("tree shape =", self.tree.shape)
            print ("tree as a matrix:")
            # Create a dataframe from tree for a user-friendly view
            df_tree = pd.DataFrame(self.tree,
                                   columns=["factor", "split val", "left", "right"])
            df_tree.index.name = "node"
            print (df_tree)
        else:
            print ("Tree has no data")
        print ("")


if __name__ == "__main__":
    print("called")