import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import joblib
from z_score import z_score_test_real_use
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,recall_score,precision_recall_curve,auc,precision_score




def final_work(input_data_path,path_mean_std,model_path,output_path):
    input_data = input_data_path 
    input_data_df = pd.read_csv(input_data)
    try:
        input_data_df = input_data_df.drop(columns=['Unnamed: 0'])  
    except:
        pass
    test_set = z_score_test_real_use(path_mean_std=path_mean_std,path_test = input_data,output=False)
    test_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = input_data_df[['HBpzpscore','EOSpzpscore','PLTpzpscore']] # replace the data that is already z score
    model = joblib.load(model_path) # import models
    pred_final = model.predict(test_set) # evaluate predict result 
    input_data_df['predict result'] = pred_final # add predict result into origin data
    input_data_df.to_csv(output_path) # output the result as .csv file

    # real_data = pd.read_csv('D:/CODE/KD/datas/KeptGPT/k-fold/30/total-test_2_z.csv') # for test the result
    # real_class = real_data['Class'] # for test the result
    # print(recall_score(real_class,input_data_df['predict result'],pos_label = 0)) # for test the result

    
if __name__ == '__main__':
    input_data_path = input('please input your data path:') #input data,must save by .csv file,and have same columns (符號必須轉換成字母p，例如seg (%) -> segp，相連的特殊符號只會轉成一個字母p)
    path_mean_std = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/2-fold_train_mean_std.csv' # input the data that use to normalized the test set
    model_path = 'D:/CODE/KD/codes/models_output/設定oversample參數型/KeptGPT/lgbm/2-model' # import models
    output_path = 'D:/CODE/KD/datas/KeptGPT/k-fold/pred_result.csv' #set the output path 
    final_work(input_data_path = input_data_path, path_mean_std = path_mean_std, model_path = model_path, output_path = output_path)
    # D:\CODE\KD\datas\KeptGPT\k-fold\30\total-test_noclass.csv