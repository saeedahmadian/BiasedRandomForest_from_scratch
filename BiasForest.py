from RandomForest import RandomForest
import numpy as np




class BRAF(object):
    def __init__(self,S,p,k,weights,name='BRAF'):
        """
        :param raw_data: specify the name of the csv file
        :param S: Spesify the size of the Biased Random Forest method
        :param p: Specify the ratio between R1 and R2
        :param k: Specify the KN Nearest Neighbours for minority class
        """
        self.S = S
        self.p = p
        self.k = k
        self.name= name
        self.weights= weights
        "Initialize the Forests"
        self.R1 = RandomForest('R1_Forest',self.weights,int(self.p*self.S),True)
        self.R2 = RandomForest('R2_Forest',self.weights, int((1-self.p) * self.S),True)

    def fit(self,data):
        """
        :param data: Read Data and preprocess for further analysis
        T is for Vanilla Random Forest and Tc is for biased forest
        :return: fitted R1 and R2
        """
        if data is not None:
            T, Tc = data
            print('fitting Biased Random Forest starts...')
            self.R1.fit(T[:,0:-1],T[:,-1].astype(np.int))
            self.R2.fit(Tc[:,0:-1],Tc[:,-1].astype(np.int))

        else:
            print("Data Not Found. Please check the file name and directory")

    def predict(self,x_test):
        """
        :param x_test: receives the given x to predict
        :return: logits
        """
        pred1 = self.R1.predict(x_test)
        pred2 = self.R2.predict(x_test)
        return (pred1+pred2)/2



