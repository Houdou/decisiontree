import numpy as np
from collections import deque

class DecisionTreeNode:
    import itertools
    nid_counter = itertools.count()

    def __init__(self, tree, sample_mask, available_attr, parent=None):
        self.tree = tree
        self.sample_mask = sample_mask
        self.available_attr = available_attr
        self.split_attr = None
        self.parent = parent

        self.uuid = next(self.nid_counter)

        self.leaf_class = -1
        self.classify_dict = {}

        # Get samples from data set
        # print("Node", self.uuid)
        self.y = self.tree.y[sample_mask]
        # print("X", self.X)
        # print("y", self.y)

        # Calculate entropy
        label_count = np.bincount(self.y)
        label_count = label_count[np.where(label_count > 0)] # remove zero sample class
        freqs = label_count / len(self.y)
        self.label_count = label_count
        self.entropy = np.sum(-freqs * np.log2(freqs))
        # print("label count", self.label_count)
        # print(self.uuid, " entropy ", self.entropy)

    def split(self, attr_index):
        # nodes = []
        split_dict = {}
        child_available_attr = self.available_attr[np.where(self.available_attr != attr_index)]
        X = self.tree.X[self.sample_mask]

        for (attr, bin) in self.tree.attr_dicts[attr_index].items():
            # Find samples belong to this attr
            mask = np.where(X[:, attr_index] == bin)[0]
            if mask.shape[0] == 0: # Empty attribute
                continue

            mask = self.sample_mask[mask]
            # print("mask", mask)
            sub_node = DecisionTreeNode(self.tree, mask, child_available_attr, self)
            split_dict[bin] = sub_node

        return split_dict

    def find_best_split(self):
        print("Sample indices", self.sample_mask + 1)
        if self.label_count.shape[0] == 1:
            print("All samples are in same label.")
            print("Mark as leaf node with label: " + self.tree.label_bin[self.y[0]])
            self.leaf_class = self.y[0]
            return -1, None

        if self.available_attr.shape[0] == 0:
            print("No avaiable splitting attribute")
            self.leaf_class = np.argmax(self.label_count)
            return -1, None

        print("Find best split on node", self.uuid)
        # print("Available attr", self.available_attr)

        max_gain = -99999
        best_split_attr = -1
        best_splits = None
        for attr_index in self.available_attr:
            # print("Split on attr", attr_index)
            split = self.split(attr_index)
            num_samples = self.sample_mask.shape[0]

            sub_node_entropy = 0.0
            for sub_node in split.values():
                num_subnode_sample = sub_node.sample_mask.shape[0]
                freq = num_subnode_sample / num_samples
                sub_node_entropy += freq * sub_node.entropy

            gain = self.entropy - sub_node_entropy
            # print("Gain: ", gain)
            if gain > max_gain:
                max_gain = gain
                best_split_attr = attr_index
                best_splits = split

        print("best split on attribute No.", best_split_attr)
        self.split_attr = best_split_attr
        self.classify_dict = best_splits
        return best_split_attr, best_splits

    def classify(self, X):
        if self.leaf_class == -1:
            return self.classify_dict[X[self.split_attr]].classify(X)
        else:
            return self.leaf_class

class DecisionTree:
    def __init__(self, X, y):
        # Record shapes
        self.num_samples, self.num_attr = X.shape

        # Allocate array for covert the attribute to bins
        self.X = np.empty((self.num_samples, self.num_attr), dtype=np.int32)
        self.y = np.empty(self.num_samples, dtype=np.int32)

        # Record the attribute-bin mapping
        # Find bin knowing attr
        self.attr_dicts = []
        self.label_dict = {}
        # Find attr knowing bin
        self.attr_bins = []
        self.label_bin = {}

        # Convert to bin
        for attr_index in np.arange(self.num_attr):
            bin = 0
            label_bin = 0
            attr_bin = {}
            attr_dict = {}
            for i in np.arange(self.num_samples):
                attr = X[i, attr_index]
                if attr not in attr_dict:
                    attr_dict[attr] = bin
                    attr_bin[bin] = attr
                    bin += 1

                self.X[i, attr_index] = attr_dict[attr]

                label = y[i]
                if label not in self.label_dict:
                    self.label_dict[label] = label_bin
                    self.label_bin[label_bin] = label
                    label_bin += 1

                self.y[i] = self.label_dict[label]

            self.attr_dicts.append(attr_dict)

        self.root = DecisionTreeNode(self, np.arange(self.num_samples), np.arange(self.num_attr))
        node_list = deque([self.root])
        while len(node_list) > 0:
            to_split = node_list.popleft()
            print("\nTo split node", to_split.uuid)
            split_attr, split_dict = to_split.find_best_split()
            if split_attr != -1:
                for node in split_dict.values():
                    # print("Add node ", node.uuid)
                    node_list.append(node)

    def classify(self, datas):
        # Convert attribute to bin
        X = np.empty((datas.shape[0], self.num_attr))
        for data in np.arange(datas.shape[0]):
            for attr_index in np.arange(self.num_attr):
                attr = datas[data, attr_index]
                bin = self.attr_dicts[attr_index][attr]
                if bin is not None:
                    X[data, attr_index] = self.attr_dicts[attr_index][datas[data, attr_index]]
                else:
                    print("Invalid attribute for", self.attr_dicts[attr_index], attr)
                    X[data, attr_index] = 0


        for x in X:
            y = self.root.classify(x)
            print(self.label_bin[y])

if __name__ == "__main__":

    # ID, {attr A, B, C, D, E, F}, label
    sampleData = np.array([
        [0, 'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'Pos'],
        [1, 'A2', 'B1', 'C2', 'D1', 'E1', 'F1', 'Pos'],
        [2, 'A2', 'B1', 'C1', 'D1', 'E1', 'F1', 'Pos'],
        [3, 'A1', 'B1', 'C2', 'D1', 'E1', 'F1', 'Pos'],
        [4, 'A3', 'B1', 'C1', 'D1', 'E1', 'F1', 'Pos'],
        [5, 'A1', 'B2', 'C1', 'D1', 'E2', 'F2', 'Pos'],
        [6, 'A2', 'B2', 'C1', 'D2', 'E2', 'F2', 'Pos'],
        [7, 'A2', 'B2', 'C1', 'D1', 'E2', 'F1', 'Pos'],
        [8, 'A2', 'B2', 'C2', 'D2', 'E2', 'F1', 'Neg'],
        [9, 'A1', 'B3', 'C3', 'D1', 'E3', 'F2', 'Neg'],
        [10, 'A3', 'B3', 'C3', 'D3', 'E3', 'F1', 'Neg'],
        [11, 'A3', 'B1', 'C1', 'D3', 'E3', 'F2', 'Neg'],
        [12, 'A1', 'B2', 'C1', 'D2', 'E1', 'F1', 'Neg'],
        [13, 'A3', 'B2', 'C2', 'D2', 'E1', 'F1', 'Neg'],
        [14, 'A2', 'B2', 'C1', 'D1', 'E2', 'F2', 'Neg'],
        [15, 'A3', 'B1', 'C1', 'D3', 'E3', 'F1', 'Neg'],
        [16, 'A1', 'B1', 'C2', 'D2', 'E2', 'F1', 'Neg'],
    ])
    # ID, {attr age, income, student, credit}, label
    sampleData = np.array([
        [0, 'youth', 'high', 'no', 'fair', 'no'],
        [1, 'youth', 'high', 'no', 'excellent', 'no'],
        [2, 'middle_aged', 'high', 'no', 'fair', 'yes'],
        [3, 'senior', 'medium', 'no', 'fair', 'yes'],
        [4, 'senior', 'low', 'yes', 'fair', 'yes'],
        [5, 'senior', 'low', 'yes', 'excellent', 'no'],
        [6, 'middle_aged', 'low', 'yes', 'excellent', 'yes'],
        [7, 'youth', 'medium', 'no', 'fair', 'no'],
        [8, 'youth', 'low', 'yes', 'fair', 'yes'],
        [9, 'senior', 'medium', 'yes', 'fair', 'yes'],
        [10, 'youth', 'medium', 'yes', 'excellent', 'yes'],
        [11, 'middle_aged', 'medium', 'no', 'excellent', 'yes'],
        [12, 'middle_aged', 'high', 'yes', 'fair', 'yes'],
        [13, 'senior', 'medium', 'no', 'excellent', 'no']
    ])

    tree = DecisionTree(sampleData[:, 1:5], sampleData[:, 5])

    test = np.array([
        ['youth', 'high', 'yes', 'fair']
    ])
    tree.classify(test)
