import numpy as np
from Node import Node
from graphviz import Graph
from Config import args


class DecisionTree(object):
    def __init__(self,name='DecisionTree',weights=[.5,1.5]):
        self.total_depth = args.depth
        self.metric = args.metric
        self.name = name
        self.weights= weights
        self.graph = Graph(name=name,format='png')
        self.total_loss = 0

    def score(self,y):
        """
        :param y: all y values
        :return: calculate the gini or entropy for any node
        """
        n_total= y.size
        num_y_per_label= [sum(y==label) for label in range(self.num_labels)]
        if self.metric == 'gini':
            return 1-sum([(num_y_per_label[i]/n_total)**2 for i in range(len(num_y_per_label))])
        else :
            return sum(list(map(lambda x: - (x/n_total)*np.log2(x/n_total) if x!=0 else 0,num_y_per_label)))

    def split(self, x, y, metric='gini'):
        """
        :param x: are the features
        :param y: are labels
        :param metric: it is 'gini' or 'entropy'
        :return:
        """
        "Specify distinct classes"
        n_total = y.size
        "specify how many number of y we have in each class"
        y_per_class = [np.sum(y == c) for c in range(self.num_labels)]

        """Specify initial entropy or gini, 
            Initial split_index and Initial split_value"""
        best_entropy_or_gini = 10
        best_split_idx = None
        best_split_value = None

        "iterate over features"
        for idx in range(x.shape[1]):
            "I slice column with index = idx and corresponding y and sort them to save time"
            col, y_label = zip(*sorted(zip(x[:, idx], y)))

            """Specify how many labels do we have in the left_split
                and since we start from the top to go down we consider we have no class
                at the begining for left hand side split
            """
            y_left_class = [0 for _ in y_per_class]

            "Instead we have every classes in right hand split and I use copy to save the original list"
            y_right_class = y_per_class.copy()

            for i in range(1, n_total):
                "Move y one by one to left hand split "
                y_left_class[y_label[i - 1]] += 1
                y_right_class[y_label[i - 1]] -= 1

                if metric == 'gini':
                    """claculate gini of left and right hand side splits"""
                    sum_=sum(self.weights)
                    y_left_gini = 1.0 - sum(
                        (self.weights[c]/sum_)*(y_left_class[c] / i) ** 2 for c in range(self.num_labels)
                    )
                    y_right_gini = 1.0 - sum(
                        (self.weights[c]/sum_)*(y_right_class[c] / (n_total - i)) ** 2 for c in range(self.num_labels)
                    )
                    "calculate total weighted gini of the split"
                    loss = (i / n_total) * y_left_gini + ((n_total - i) / n_total) * y_right_gini
                else:
                    "find the probability of each class within each split"
                    sum_weights=sum(self.weights)
                    sum_left=sum(y_left_class)
                    sum_right=sum(y_right_class)
                    y_left_prob = [(yy / sum_left)*(w/sum_weights) for yy,w in zip(y_left_class,self.weights)]
                    y_right_prob = [(yy / sum_right)*(w/sum_weights) for yy,w in zip(y_right_class,self.weights)]

                    "claculate entropy for left hand split and right hand split"
                    entropy_left = sum(list(map(lambda x: -x * np.log2(x) if x != 0 else 0, y_left_prob)))
                    entropy_right = sum(list(map(lambda x: -x * np.log2(x) if x != 0 else 0, y_right_prob)))

                    "calculate total weighted entropy"
                    loss = ((i) / n_total) * entropy_left + ((n_total - i) / n_total) * entropy_right

                "ignore the same features in a column"
                if col[i - 1] == col[i]:
                    continue
                """if current loss is less than the minimum loss (entropy or gini), then change their place as well as 
                    corresponding column index and the mean average value of current and next values"""
                if loss < best_entropy_or_gini:
                    best_entropy_or_gini = loss
                    best_split_idx = idx
                    best_split_value = (col[i - 1] + col[i]) / 2
        return [best_entropy_or_gini, best_split_value, best_split_idx]


    def create_tree(self,x,y,depth=0,name='root'):
        """
        :param x: training features
        :param y: training labels
        :param depth: depth of three
        :return: returns a tree of nodes (expand root node to the leafs)
        I used pre-ordered traversal to create tree root-->left-->right
        """
        """
        Initialize the current_node
        """
        num_y_per_label= [sum(y==label) for label in range(self.num_labels)]
        current_node = Node(score=self.score(y),depth= depth,num_y_per_label=num_y_per_label,name=name)

        "if we want to save the graph of current tree"
        """
        Specify the stopping situation of expanding nodes it is eighter total depth or gini-entropy= is equal zero
        the second condition prevents from over fitting
        """
        if depth < self.total_depth and current_node.IsLeaf()==False :
            "get the best place to split for current node"
            curr_loss, cur_split_value, cur_split_idx = self.split(x,y,self.metric)
            "make sure I have some output to update"
            if cur_split_value != None:
                "update the current node with new values"
                current_node.score= curr_loss
                current_node.split_index= cur_split_idx
                current_node.split_feature =cur_split_value
                if current_node.depth == self.total_depth-1:
                    self.total_loss += current_node.score *(sum(current_node.num_y_per_label)/self.sample_size)
                "Specify the bounds to split the current node to left and right "
                boundry = x[:,cur_split_idx] < cur_split_value
                "create left and right nodes using recursive method and each iteration x,y,depth and node position"
                current_node.left= self.create_tree(x[boundry],y[boundry],depth+1,'left_{}_{}'.format(depth+1,current_node.name))

                current_node.right= self.create_tree(x[~boundry],y[~boundry],depth+1,'right_{}_{}'.format(depth+1,current_node.name))

        return current_node

    def fit(self,x,y):
        """
        :param x: x_train
        :param y: y_train
        :return: returns the created tree
        """
        print('Tree {} starts fitting'.format(self.name))
        self.num_labels= len(set(y))
        self.sample_size = y.size
        self.decision_tree = self.create_tree(x,y)
        if args.draw_trees == True:
            print("Corresponding Graphs of trees are saved in {}".format(args.dir))
            graph=self.squized_graph(self.decision_tree,{})
            self.draw_graph(graph)
            self.graph.render('{}.gv'.format(self.name),args.dir)

    def squized_graph(self,node,dic={}):
        """
        :param node: start from given node (which in our case is root)
        :param dic: initial dictionary to gather information for drawing
        :return: dictionary of graph
        """
        if node.left != None:
            dic[node]=[node.left,node.right]
            self.squized_graph(node.left,dic)
            self.squized_graph(node.right, dic)
        return dic

    def draw_graph(self,data):
        """
        :param data: dictionary graph of the network
        :return: graph of  network in form of .png or .pdf
        """
        "list of all nodes"
        list_nodes=list(data.keys())
        for node in list_nodes:
            self.graph.node(node.name,'depth : {} \n Score {} : {} \n y per sample:{}'.
                            format(node.depth,self.metric,node.score,node.num_y_per_label))
        """
        connected nodes together if they are not in the initial list 
        means they are leaf and we create them
        """
        for key,value in data.items():
            if value[0] not in list_nodes:
                list_nodes.append(value[0])
                self.graph.node(value[0].name,'depth : {} \n Score {} : {} \n y per sample:{} '.
                            format(value[0].depth,self.metric,value[0].score,value[0].num_y_per_label))
            if value[1] not in list_nodes:
                list_nodes.append(value[1])
                self.graph.node(value[1].name,'depth : {} \n,Score {} : {} \n y per sample:{}'.
                            format(value[1].depth,self.metric,value[1].score,value[1].num_y_per_label))
            self.graph.edge(key.name,value[0].name,'x[{}] < {}'.
                            format(key.split_index,key.split_feature))
            self.graph.edge(key.name,value[1].name,'x[{}] > {}'.
                            format(key.split_index,key.split_feature))

    def predict(self,x):
        """
        :param x: x test
        :return: predicted label for given x
        """
        "Then we create and empty prediction list"
        predicted_labels= []
        "For each sample we walk through the tree and if we reach to leaf we do prediction"
        for x_i in x:
            """
            first we get the created tree from fit method and initialize a tree with fitted one
            """
            fitted_tree = self.decision_tree

            """
            loop through tree to reach the leaf node
            """
            while(fitted_tree.left):
                """
                start from left node if it is selected proceed with that
                else choose the right node
                """
                if x_i[fitted_tree.split_index] < fitted_tree.split_feature:
                    fitted_tree = fitted_tree.left
                else:
                    fitted_tree= fitted_tree.right
            """
            attache the predicted labels to list
            """
            predicted_labels.append(np.argmax(fitted_tree.num_y_per_label))
        return np.array(predicted_labels)

















