from Decision_tree import DecisionTree
import numpy as np

class RandomForest(object):
    def __init__(self, name="RandomForest",class_weights=[.2,.8],forest_size=10,bootsrap=True):
        """
        :param name: Initialize the Forest with its name
        Also with number of trees and Initialize trees
        """
        self.name = name
        self.num_trees= forest_size
        self.weights = class_weights
        self.trees= [DecisionTree('DecisionTree_'+str(i),weights=self.weights) for i in range(self.num_trees)]
        self.bootstrap = bootsrap


    def fit(self,x,y):
        """Bootstrap sampling for number of trees in the forest
        based on the paper we will have 64 % of training data in each tree and
        replacement is OK means that we can have repetitive samples
        if bootsrap sampling is false then we seerate data and feed it to different trees
        """
        print('Forest {} starts fitting {} trees'.format(self.name,self.trees))
        if self.bootstrap == True:
            self.sample_size = int(y.size*.64)
        else:
            self.sample_size = int(y.size /self.num_trees)

        "Create the container (DataLoader plays the rule  of DataLoader in tensorflow) for trees "
        DataLoader = []
        for i in range(self.num_trees):
            indx= np.random.permutation(self.sample_size)
            DataLoader.append([x[indx,:],y[indx]])
        c=0
        """
        fit each tree in the forest
        """
        for x_tree,y_tree in DataLoader:
            self.trees[c].fit(x_tree,y_tree)
            c+=1

    def predict(self,x):
        """
        :param x: input data
        :return: predicted value (logits)
        """
        return np.mean(np.array([tree.predict(x) for tree in self.trees]),axis=0)


