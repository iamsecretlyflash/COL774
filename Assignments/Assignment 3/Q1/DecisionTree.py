import numpy as np
from collections import defaultdict
class TreeNode:

    __slots__ = ['depth', 'children', 'is_leaf', 'value', 'split_index', 'median', 'info_gain']
    def __init__(self, depth, info_gain = None, is_leaf = False, value = 0, split_index = None):

        #to split on column
        self.depth = depth

        #add children afterwards
        self.children = dict()

        #if leaf then also need value
        self.value = None
        self.info_gain = info_gain
        self.is_leaf = is_leaf
        self.median = 0
        if(self.is_leaf):
            self.value = value
        
        self.split_index = None
        if(not self.is_leaf):
            self.split_index = split_index

    def __repr__(self):
        '''
        Returns: string representation of the node
        '''
        return str(self)

    def __str__(self):

        return f'Depth = {self.depth} | IS_LEAF : {self.is_leaf} and value : {self.value} | SPLIT ON : {self.split_index} and GAIN : {self.info_gain}'


class DecisionTree:

    __slots__ = ['root', 'max_depth', 'feature_types', 'node_level_dict', 'prunes']
    def __init__(self):
        '''
        Constructor
        '''
        self.root = None  
        self.max_depth = 0 # max_Depth of the tree
        self.node_level_dict = defaultdict(lambda : []) # to store the nodes at each depth
    
    def entropy(self, data : np.ndarray):
        '''
        Calculates the entropy of the given data

        Args: data: numpy array of shape [num_example, num_features + 1] : Last column is the target
        Returns the entropy of the data
        '''
        _, counts = np.unique(data[:,-1], return_counts= True)
        probs = counts/data.shape[0] 
        return -np.sum(probs * np.log2(probs))

    def information_gain(self, data : np.ndarray, attr_index : int):
        '''
        Calculates the information gain of the attribute at attr_index using the given data

        Args: data: numpy array of shape [num_example, num_features + 1] : Last column is the target
              attr_index: index of the attribute to split on and calculate the information gain
        '''
        mutual_info = self.entropy(data) # H(Y)

        if self.feature_types[attr_index] == 1: # categorical

            attr_vals, attr_counts = np.unique(data[:,attr_index], return_counts = True)
            attr_probs = attr_counts / data.shape[0]
            for i in range(len(attr_vals)):
                mutual_info -= attr_probs[i] * self.entropy(data[data[:,attr_index] == attr_vals[i]])
        else : # continuous

            median_val = np.median(data[:,attr_index]) # find median and plit on median
            left = data[data[:,attr_index] <= median_val]
            right = data[data[:,attr_index] > median_val]
            mutual_info = mutual_info - ((len(left) * self.entropy(left)) + (len(right) * self.entropy(right)))/ len(data)

        return mutual_info

    def build_tree(self, data : np.ndarray, features : list, depth : int):
        '''
        Recursively build the decision tree using the given data and features

        Args:        data: numpy array of shape [num_example, num_features + 1] : Last column is the target
                 features: list of indices of available attributes to split on
                    depth: depth of the node in the tree
                
                  Returns: root of the decision tree'''

        val = 1 if np.mean(data[:,-1]) >= 0.5 else 0
        if depth == self.max_depth or len(np.unique(data[:,-1])) == 1: # last depth. predict the most common
            node = TreeNode(depth, is_leaf = True, value = val)
            self.node_level_dict[depth].append(node)
            return node
        
        best_attr = None
        best_gain = -float('inf')
    
        for attr_index in features : #finding the best attribute with max informatoin gain
            gain_local = self.information_gain(data, attr_index)
            if gain_local > best_gain :
                best_gain = gain_local
                best_attr = attr_index

        node = TreeNode(depth=depth, is_leaf=False, split_index=best_attr)
        node.info_gain = best_gain
        node.value = val

        if self.feature_types[best_attr] == 1:
            attr_vals = np.unique(data[:,best_attr])
            for attr in attr_vals : #adding children corresponding to each attribute value
                node.children[attr] = self.build_tree( data[data[:, best_attr] == attr], features, depth +1)
        else :
            median_val = np.median((data[:,best_attr]))
            node.median = median_val
            left = data[ data[:,best_attr] <= median_val]
            right = data[ data[:,best_attr] > median_val]
            if len(left) :
                node.children[0] = self.build_tree(left, features, depth+1)
            if len(right):
                node.children[1] = self.build_tree(right, features, depth+1)

        self.node_level_dict[depth].append(node)
        return node

    def fit(self, X, y, types ,max_depth):
        '''
        Fits a decision tree to X and y

        Args:       X: numpy array of data [num_samples, num_features]
                    y: numpy array of labels [num_samples, 1]
                types: list of types of features (0: continuous, 1: categorical)
                max_depth: maximum depth of the tree
        
        '''
        self.max_depth = max_depth
        self.feature_types = types
        self.root = self.build_tree(np.concatenate([X,y], axis = 1), list(range(len(types))), 0)

    def recurive_predict(self, node, X):
        '''
        Recursively predicts the value for the given node and data

        Args: node: node of the tree
                 X: numpy array of data [num_samples, num_features]

        Returns: predicted value
        '''
        
        if node.is_leaf:
            return node.value
        if self.feature_types[node.split_index] == 1:
            child_val = int(X[node.split_index])
            if child_val not in node.children:
                return node.value
            return self.recurive_predict(node.children[child_val], X)
        else :
            if X[node.split_index] <= node.median:
                if 0 not in node.children:
                    return node.value
                return self.recurive_predict(node.children[0], X)
            else :
                if 1 not in node.children:
                    return node.value
                return self.recurive_predict(node.children[1], X)

    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        return np.array([self.recurive_predict(self.root, x) for x in X])
    
    def post_prune(self, X_val, y_val, X_train, y_train, X_test, y_test):
        '''
        Post prunes the decision tree using the given validation data using a greedy approach
        Start trimming all nodes at the last level and move up the tree

        Args: X_val: numpy array of data [num_samples, num_features] for validation
                y_val: numpy array of labels [num_samples, 1] for validation
                X_train: numpy array of data [num_samples, num_features] for training
                y_train: numpy array of labels [num_samples, 1] for training
                X_test: numpy array of data [num_samples, num_features] for testing
                y_test: numpy array of labels [num_samples, 1] for testing

        Returns: val_acc_list: list of validation accuracies after each prune
                    train_acc: list of training accuracies after each prune
                     test_acc: list of testing accuracies after each prune
        '''
        val_best = np.mean(self(X_val) == y_val.T[0])
        val_acc_list = []; train_acc = []; test_acc = []
        print(f'Vaidation Accuracy = {val_best}')
        self.prunes = defaultdict(list)
        net_trimmed =0
        for level in range(self.max_depth - 1, -1, -1):
            node = self.node_level_dict[level]
            for n in node:
                if n.is_leaf == False:
                    n.is_leaf = True
                    val_acc = np.mean(self(X_val) == y_val.T[0])
                    if val_acc > val_best:
                        val_best = val_acc
                        val_acc_list.append(val_best)
                        train_acc.append(np.mean(self(X_train) == y_train.T[0]))
                        test_acc.append(np.mean(self(X_test) == y_test.T[0]))
                        self.prunes[level].append(n)
                        net_trimmed += 1
                    else: n.is_leaf = False
        print(f'Vaidation Accuracy = {val_best} | Total nodes trimmed = {net_trimmed}')
        return val_acc_list, train_acc, test_acc, net_trimmed
