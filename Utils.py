import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import pickle
from Config import args
import os

class Binary_metrics(object):
    def __init__(self,y_true,logits):
        """
        This class measures the classification metrics
        :param y_true: are the true probabilities
        :param logits: are predicted probabilities by model
        """
        "Calculate True positive, True negative, False positive and False negative"
        self.TP = np.sum((logits >=.5) & (y_true==1))
        self.TN= np.sum((logits<.5) & (y_true==0))
        self.FP = np.sum((logits >= .5) & (y_true == 0))
        self.FN = np.sum((logits <.5) & (y_true==1))
        self.auc = roc_auc_score(y_true,logits)

    def precision(self):
        return (self.TP)/(self.TP+self.FP)

    def recall(self):
        return self.TP/(self.TP+self.FN)

    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)

    def f1_score(self):
        return 2*self.precision()*self.recall()/(self.precision()+self.recall())

    def false_positive_rate(self):
        return (self.FP)/(self.FP+self.TN)

def k_Nearest_Neigh(t_i,T_maj,k):
    """
    :param t_i: minority sample
    :param T_maj: Majority data set
    :param k: number of nearest neighbours
    :return: return KNN
    """
    x_i= t_i[0:-1]
    x_maj= T_maj[:,0:-1]
    return T_maj[np.argsort(np.sqrt(np.sum((x_i-x_maj)**2,axis=-1)))][0:k]

def Split_Maj_Min(data):
    """
    :param data: receives data set
    :return: splits the majority and minority data sets
    """
    "calculate the shape of each groups and returns majority and minority respectively"
    "T0 is the data with label 0"
    "T1 is the data with label 1"
    T0= data[data[:,-1]==0]
    T1= data[data[:,-1]==1]
    if T0.shape[0] !=0 and T1.shape[0] !=0:
        if T0.shape[0] > T1.shape[0]:
            return T0,T1
        else:
            return T1,T0
    else:
        print('one or two groups are empty')
        return False

def critical_area(Tmaj,Tmin,k):
    """
    :param Tmaj: Majority dataset
    :param Tmin: Minority dataset
    :param k: number of KNN
    :return: Critical area
    """
    Tc=[]
    for t_i in Tmin:
        Tc.append(list(t_i))
        Tnn= k_Nearest_Neigh(t_i,Tmaj,k)
        for tj in Tnn:
            if list(tj) not in Tc:
                Tc.append(list(tj))

    return np.array(Tc)

def preprocess(file_name='Data/diabetes.csv'):
    """
    :param file_name: input csv file
    :return: cleaned data
    """
    df = pd.read_csv(file_name,header=0).interpolate('linear')
    return df.values



def Kfold_cross_validation(k,data,main_model,name='BRAF'):
    """
    :param k: number of folds
    :param data: data set including T and Tc
    :param main_model: Biased Forest class
    :param name: name of model
    :return:
    """
    T= data[0]
    Tc= data[1]
    "Initialize dictionary for cross validation"
    history = {}
    history['Accuracy'] = []
    history['Recall'] = []
    history['F1_score'] = []
    history['Precision'] = []
    history['False Positive rate'] = []
    history['AUC']=[]
    "Specify length of T and Tc chunks for cross validation"
    batch_size_T = int(T.shape[0]/k)
    batch_size_Tc= int(Tc.shape[0]/k)
    for i in range(k):
        print('{}th fold starting'.format(k))
        idx_test_T = [j for j in range(i*batch_size_T,(i+1)*batch_size_T)]
        idx_train_T= [j for j in range(0,i*batch_size_T)]+[j for j in range((i+1)*batch_size_T,(k)*batch_size_T)]
        idx_test_Tc = [j for j in range(i * batch_size_Tc,(i + 1) * batch_size_Tc)]
        idx_train_Tc= [j for j in range(0,i*batch_size_Tc)]+[j for j in range((i+1)*batch_size_Tc,(k)*batch_size_Tc)]
        T_test = T[idx_test_T]
        T_train= T[idx_train_T]
        Tc_test = T[idx_test_Tc]
        Tc_train = T[idx_train_Tc]
        "Draw Trees of the model for the last cross validation"
        if i == k-1:
            args.draw_trees = True

        model = main_model
        model.fit([T_train, Tc_train])
        data_test = np.concatenate([T_test, Tc_test], axis=0)
        x_test = data_test[:, 0:-1]
        y_test = data_test[:, -1]
        logits = model.predict(x_test)
        metrics = Binary_metrics(y_test, logits)
        history['Accuracy'].append(metrics.accuracy())
        history['Recall'].append(metrics.recall())
        history['F1_score'].append(metrics.f1_score())
        history['Precision'].append(metrics.precision())
        history['False Positive rate'].append(metrics.false_positive_rate())
        history['AUC'].append(metrics.auc)
        "Save the figures for ROC_AUC and PCR"
        plot_ROC_PRC(y_test, logits,metrics.accuracy(),metrics.precision(),metrics.recall(),
                     name='ROC_ORC_{} th _fold'.format(i), save_dir='curves')
        save_model(model, model.name + str(i), dir='saved_models')
        print('Accuracy {}, precision {}, recall {}, AUC {}, for {}th fold'.
              format(metrics.accuracy(),metrics.precision(),metrics.recall(),metrics.auc,i))
    pd.DataFrame(history).to_csv(args.result_dir_name+'/'+name+'.csv')

def plot_ROC_PRC(y_true,logits,acc,precis,rec,name='ROC_ORC',save_dir=args.save_fig_dir):
    """
    plots the ROC and PRC curves and save them as well
    :param y_true: are the true probabilities
    :param logits: are the predicted logits
    :param acc: accuracy of current fold
    :param precis: precision of the current fold
    :param rec: recall for the current fold
    :param name: name of the graph to save
    :param save_dir: direction to save the model
    :return: saves the figures
    """
    plt.figure(figsize=(15,12))
    fig,ax = plt.subplots(1,2,sharex='none',sharey='none')
    FPR, TPR, _ = roc_curve(y_true, logits)
    precision, recall, _ = precision_recall_curve(y_true, logits)
    ax[0].plot(FPR,TPR,color='maroon',
               label='ROC curve \n fold_Acc= {} \n fold_precision {} \n fold_recall {}'.
               format(acc,precis,rec))
    ax[0].plot(FPR, FPR,linestyle='--', color='blue',
               label='random selection')
    ax[0].legend()
    ax[0].set_xlabel('False Poitive rate')
    ax[0].set_ylabel('True Poitive rate (recall)')
    ax[1].plot(recall,precision,color='lime',label='PRC ')
    ax[1].legend()
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    fig.tight_layout()
    fig.savefig(save_dir+'/'+name+'.png')
    fig.show()

def save_model(model,model_name,dir=args.save_models_dir):
    """
    :param model: Braf model to save
    :param model_name: name of braf model
    :param dir:
    :return:
    """
    file_handler = open(dir+'/'+model_name+'.obj', 'wb')
    pickle.dump(model,file_handler)

def load_model():
    """
    :return: returns the last saved model
    """
    if os.listdir(args.save_models_dir)[-1] !=None:
        filehandler = open(os.listdir(args.save_models_dir)[-1], 'rb')
        return pickle.load(filehandler)
    else:
        print("No saved model")
        return False














