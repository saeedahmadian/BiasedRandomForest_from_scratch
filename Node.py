class Node(object):
    def __init__(self, score,depth,num_y_per_label,name):
        """
        :param score: is the entropy or gini value of the node
        :param depth: the depth of the current node in the tree
        :param num_y_per_label : number of y per label for current node
        """
        self.score = score
        self.depth = depth

        """
        left and right node for the current node
        """
        self.left = None
        self.right = None

        """
        split index and features for the current node
        """
        self.split_index = None
        self.split_feature = None

        """
        total samples and number of y per label and predicted class of
        """
        self.num_y_per_label = num_y_per_label

        "Specify whether this node is root or left or right"
        self.name= name

    def IsLeaf(self):
        if self.score==0:
            return True
        else:
            return False
