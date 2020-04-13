import argparse

parser = argparse.ArgumentParser('Biased Random Forest')
parser.add_argument("--model_name", default='BRAF', help="Specify forest size",
                    type=str)
parser.add_argument("--S", default=100, help="Specify forest size",
                    type=int)
parser.add_argument("--weights", default=[1,1], help="Specify class weights",
                    type=list)
parser.add_argument("--p", default=.5, help="Specify p ratio",
                    type=float)
parser.add_argument("--k", default=10, help="Specify K nearest neighbours for critical area",
                    type=float)
parser.add_argument("--depth", default=5, help="Specify depth of trees",
                    type=int)
parser.add_argument("--metric", default='gini',help="Specify metric as 'gini' or 'entropy' in each tree",
                    type=str)
parser.add_argument("--k_fold", default=10,help="Specify number of k-folds for cross validation",
                    type=int)
parser.add_argument("--save_fig_dir", default='curves',help="Specify direction to save the result figures",
                    type=str)
parser.add_argument("--save_models_dir", default='saved_models',help="Specify direction to save the Biased Random Forest",
                    type=str)
parser.add_argument("--csv_file_dir", default='Data/diabetes.csv',help="Specify data direction and name",
                    type=str)

parser.add_argument("--draw_trees", default=False,help="Do you want to see the trees in each forest or not",
                    type=bool)
parser.add_argument("--dir", default='tree_graphs',help="Specify dirction for drawing the graph tree",
                    type=str)
parser.add_argument("--result_dir_name", default='results',help="Specify dirction for results of Kfold",
                    type=str)

args = parser.parse_args()


