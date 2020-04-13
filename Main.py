from BiasForest import BRAF
from Config import args
from Utils import preprocess,Split_Maj_Min,critical_area,Kfold_cross_validation


def run():
    print('The predefualt values are \n {}'.format(args))
    "Initialize the model"
    wights=args.weights
    model = BRAF(args.S, args.p, args.k,wights,args.model_name)
    "Read the data"
    data = preprocess(file_name=args.csv_file_dir)
    "Specify T and Tc based on the paper"
    Tmaj, Tmin = Split_Maj_Min(data=data)
    Tc = critical_area(Tmaj, Tmin, args.k)
    "Run the cross validation and save the results"
    Kfold_cross_validation(args.k_fold, [data, Tc], model, name=args.model_name)



if __name__=='__main__':
    run()

