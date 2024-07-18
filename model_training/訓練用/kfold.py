import warnings
import json
from cProfile import label
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
# from sklearnex import patch_sklearn  ## 加速svm
# patch_sklearn("SVC")
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,recall_score,precision_recall_curve,auc,precision_score,f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample,shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from learn_curving_fold import plot_learning_curves
# from imblearn.over_sampling import RandomOverSampler
import shap
import time
import copy
#from svm-gpu import SVM
from IPython.display import display
import os
import joblib
from tqdm import trange
# from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Conv1D, AveragePooling1D, BatchNormalization
# from keras.layers.regularization.dropout import Dropout
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import regularizers

''' 版本與部分註解
1. 先將model 內的rng(step)改成np.geometric增加，之後再看要不要改回linspace
2. 訓練scikit-learn version:1.0.2 if 有問題，記得條回來(svm prob會出問題)，score_likelihood.ipynb使用 version 1.2.2 scikit-learn
'''
''' 使用檔案
Before using this module, have to seperate the data for each fold first,
    。save path and name as '/{}-fold_test_z.csv' for validation to use as single fold
    。save path and name as '/{}-fold_train_z.csv'for training to use as single fold
    (z stand for z-score if your data is not tranfer into z-score, this module would still work)
'''
''' module 細節
1. training 與 validation1部分:
			。 kfold_self:會繪製loss or f1 learning curve，不會儲存模型，但會return learning curve 細項，給定參數為 : kfold_self(model = model,point_amount=18,random_state=30,path = path,space='geomspace',loss_log=True)
當中的random_state在內部運行時，upsampling為42。
		
			。kfold_self_score : 會輸出評分參數(無oversampling)，不會儲存模型，參數方式 : kfold_self_score(path =path,model = model,scoring=['f1','precision','recall','specificity'],seperate = False,random_state=30,model_name = model_name)
		
			。 kfold_self_score_up : 會輸出評分參數(oversampling)，不會儲存模型，參數方式 : kfold_self_score_up(path = path,model = model,scoring=['f1','precision','recall','specificity'],seperate = False,random_state=30,model_name = model_name)

			。prc_plot : 繪製pr curve，並回傳相關數值，會儲存模型，參數方式 : prc_plot(path=path,random_state_1=30,model = model,model_amount=1,color_origin='orange',color_up='red',model_save_path = model_save_path)

			。roc_plot : 繪製roc curve，並回傳相關數值，不會儲存模型，參數方式 : roc_plot(path=path,random_state_1=30,model = model,model_amount=1,color_origin='orange',color_up='red',color_origin_2='blue',color_up_2='skyblue')

2. test 部分:		 
			。test_score_given_model : 由給定的模型與測試集輸出評分參數，參數方式 : test_score_given_model(model_path=model_path,file_test=file_test,random_state_test = 30)

			。test_prc_given_model : 由給定的模型與測試集輸出pr curve，參數方式 : test_prc_given_model(model_path=model_path,file_test=file_test,random_state_test = 30)

			。test_roc_given_model : 由給定的模型與測試集輸出roc curve，參數方式 : test_roc_given_model(model_path=model_path,file_test=file_test,random_state_test = 30)
'''

'''=============functions for the graphics========='''
def plot_confusion_matrix(cf_matrix):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])


def plot_roc_curve(fper,tper,fper_up,tper_up,auc,auc_up,color_origin='orange',color_up='red'):
    '''
    for i in range(len(fper)):
        plt.plot(fper[i],tper[i],color='red')#,label='ROC,auc:'+'%.3f'%(sum(auc)/len(auc)))
        plt.plot(fper_up[i],tper_up[i],color = 'orange',linestyle = '--')#,label = 'ROC_up,auc:'+'%.3f'%(sum(auc_up)/len(auc_up)))
    '''
    ## plot the ROC region for orgin distribution 
    item_max = auc.index(max(auc))
    item_min = auc.index(min(auc))
    xfill = np.sort(np.concatenate([fper[item_max], fper[item_min]]))
    y1fill = np.interp(xfill, fper[item_max],tper[item_max])
    y2fill = np.interp(xfill, fper[item_min], tper[item_min])
    plt.plot(fper[item_max],tper[item_max],color = color_origin,linestyle='--', alpha=0.6)
    plt.plot(fper[item_min],tper[item_min],color = color_origin,linestyle='--',alpha=0.6)
    plt.plot(xfill,(y1fill+y2fill)/2,color = color_origin)
    plt.fill_between(xfill, y1fill, y2fill, interpolate=True, color=color_origin, alpha=0.5)
    ## plot the ROC region for upsampling distrubution
    item_up_max = auc_up.index(max(auc_up))
    item_up_min = auc_up.index(min(auc_up))
    xfill_up = np.sort(np.concatenate([fper_up[item_up_max], fper_up[item_up_min]]))
    y1fill_up = np.interp(xfill_up, fper_up[item_up_max],tper_up[item_up_max])
    y2fill_up = np.interp(xfill_up, fper_up[item_up_min], tper_up[item_up_min])
    plt.plot(fper_up[item_up_max],tper_up[item_up_max],color = color_up,linestyle='--',alpha=0.6)
    plt.plot(fper_up[item_up_min],tper_up[item_up_min],color = color_up,linestyle='--',alpha=0.6)
    plt.plot(xfill_up,(y1fill_up+y2fill_up)/2,color = color_up)
    plt.fill_between(xfill_up,y1fill_up,y2fill_up,interpolate=True,color =color_up,alpha = 0.15)

    plt.plot(1,1,color=color_origin,label='ROC,auc:'+'%.3f'%(sum(auc)/len(auc)))
    plt.plot(1,1,color = color_up,label = 'ROC_up,auc:'+'%.3f'%(sum(auc_up)/len(auc_up)))
    plt.plot([0,1],[0,1],color = 'green',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.legend(loc='lower right')
    return(fper[item_max],tper[item_max],fper[item_min],tper[item_min],fper_up[item_up_max],tper_up[item_up_max],fper_up[item_up_min],tper_up[item_up_min],sum(auc)/len(auc),sum(auc_up)/len(auc_up))


def plot_prc_curve(precision,recall,precision_up,recall_up,auc,auc_up,color_origin='orange',color_up='red'):
    # plot the PR region for origin distribution
    item_max = auc.index(max(auc))
    item_min = auc.index(min(auc))
    print('origin_best:',item_max+1,'origin_min:',item_min+1)
    #print(len(recall[item_max]))
    ## 修正接近x軸為零時，會有(0,0)產生
    recall_min = list(recall[item_min][0:(len(recall[item_min]))])
    precision_min = list(precision[item_min][0:(len(recall[item_min]))])
    recall_max = list(recall[item_max][0:(len(recall[item_max]))])
    precision_max = list(precision[item_max][0:(len(recall[item_max]))])
    #繪圖(sort&concatenate是為了內插而產生的X點)
    xfill = np.sort(np.concatenate([recall_max, recall_min]))
    y1fill = np.interp(xfill, recall_max[::-1],precision_max[::-1]) #因為x為 1 -> 0 而非0->1，所以需要再使用np.interp的時候要先顛倒x和y，使其由小到大才能做內插
    y2fill = np.interp(xfill,recall_min[::-1], precision_min[::-1])#因為x為 1 -> 0 而非0->1，所以需要再使用np.interp的時候要先顛倒x和y，使其由小到大才能做內插
    plt.plot(recall_max,precision_max,color = color_origin,linestyle='--',alpha=0.6)
    plt.plot(recall_min,precision_min,color = color_origin,linestyle='--',alpha=0.6)
    plt.plot(xfill,(y1fill+y2fill)/2,color = color_origin)
    plt.fill_between(xfill, y2fill, y1fill,interpolate=True, alpha=0.5, color=color_origin)


    #print(*precision[item_max])
    #plot the PR region for upsampling distribution
    item_up_max = auc_up.index(max(auc_up))
    item_up_min = auc_up.index(min(auc_up))
    print('up_best:',item_up_max+1,'up_min:',item_up_min+1)
    #修正接近x軸為零時，會有(0,0)產生
    recall_min_up = list(recall_up[item_up_min][0:(len(recall_up[item_up_min]))])
    precision_min_up = list(precision_up[item_up_min][0:(len(recall_up[item_up_min]))])
    recall_max_up = list(recall_up[item_up_max][0:(len(recall_up[item_up_max]))])
    precision_max_up = list(precision_up[item_up_max][0:(len(recall_up[item_up_max]))])
    #繪圖(sort&concatenate是為了內插而產生的X點)
    xfill_up = np.sort(np.concatenate([recall_max_up, recall_min_up]))
    y1fill_up = np.interp(xfill_up, recall_max_up[::-1],precision_max_up[::-1])
    y2fill_up = np.interp(xfill_up, recall_min_up[::-1], precision_min_up[::-1])
    plt.plot(recall_max_up,precision_max_up,color = color_up,linestyle='--',alpha=0.6)
    plt.plot(recall_min_up,precision_min_up,color = color_up,linestyle='--',alpha=0.6)
    plt.plot(xfill_up,(y1fill_up+y2fill_up)/2,color = color_up)
    plt.fill_between(xfill_up,y1fill_up, y2fill_up, interpolate=True, alpha=0.5, color=color_up)

    '''
    for i in range(len(precision)):
        plt.plot(precision[i],recall[i],color='red')#,label='PRC{},auc:'.format(i+1)+'%.3f'%(auc[i])) 
        plt.plot(precision_up[i],recall_up[i],color = 'orange',linestyle = '--')#,label = 'PRC_up{},auc:'.format(i+1)+'%.3f'%(auc_up[i]))
    '''
    plt.plot(1,1,color = color_origin,label='PRC,auc(avg):'+'%.3f'%(sum(auc)/len(auc)))
    plt.plot(1,1,color = color_up,linestyle = '--',label = 'PRC_up,auc(avg):'+'%.3f'%(sum(auc_up)/len(auc_up)))
    #plt.scatter(0.925,0.05)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.legend(loc='lower left')
    return(recall_min,precision_min,recall_max,precision_max,recall_min_up,precision_min_up,recall_max_up,precision_max_up,sum(auc)/len(auc),sum(auc_up)/len(auc_up))


def roundover(x):
    if x <0:
        x = 0
    elif x>1:
        x = 1
    elif x>0 and x<1:
        x = x
    return x

'''=============functions for k-fold validaton========='''
def kfold_self(
    cv = 5,
    model = None,
    point_amount = 20,
    sub_plot = False,
    loss_log = False,
    plot =True,
    random_state  = 30,
    scoring = "misclassification error",
    path = 'k_fold_train_validation_42',
    space = 'geomspace'
    ):
    '''
    out_put -- the learning curve for average from k-fold or learning curve for each single fold
    cv : inter, default:5 -- input how many fold we want to do
    model:default: None -- input the training model 
    point_amount: inter,default:20 -- set the train_size(input as how many point want to be seperate by max length of training data)
    sub_plot :boolean,default:False  -- True if want to plot the learning curve for each fold
    loss_log: boolean,default:False -- True if want to get the loss log for each fold,
                                        False for the average loss for 
                                        -train
                                        -nonupsampling validation
                                        -upsampling validation
    plot :boolean,default:True --True if want to plot the learning for after average
    path: the folder for the datas
    '''
    rng =0
    train_loss = []
    test_loss = []
    train_loss_up = []
    test_loss_up = []
    total_train_loss_pd = pd.DataFrame()
    total_test_loss_pd = pd.DataFrame()
    total_test_loss_up_pd = pd.DataFrame()
    for i in range(1,cv+1):
        train_loss.append([])
        test_loss.append([])
        train_loss_up.append([])
        test_loss_up.append([])
    for i in trange(1,cv+1):
        #print('fold',i)
        fold_train = pd.read_csv(path+'/{}-fold_train_z.csv'.format(i))
        fold_test = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
        ##rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        '==============================================================================='   
        '''
        x_fold_train = fold_train.drop(columns=['Class'])
        y_fold_train = fold_train["Class"]
        ros = RandomOverSampler(random_state=42)
        x_train_ros, y_train_ros= ros.fit_resample(x_fold_train, y_fold_train)
        #print(sorted(Counter(y_train_ros).items()))
        #print(x_train_ros.shape,y_train_ros.shape)
        train_up = x_train_ros
        train_up['Class'] = y_train_ros
        train_up = train_up.sample(frac=1,axis =1).reset_index(drop=True)
        #print(train_up)
        # set the train set for single fold
        x_train = train_up.drop(columns=['Class'])
        y_train = train_up['Class']
        '''
        '==============================================================================='
        
        ## upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        ## fold_train = fold_train.sample(frac=1,axis = 0).reset_index(drop=True)
        fold_train = shuffle(fold_train,random_state = random_state)
        fold_train= fold_train.reset_index(drop = True)
        
        ## set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
       
        ## set the non upsampling validation for the single fold       
        fold_test = shuffle(fold_test,random_state = random_state)
        fold_test = fold_test.reset_index(drop = True)
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']
        ## log train/validationS loss for non upsampling
        train_loss[i-1],test_loss[i-1] = plot_learning_curves(x_train,y_train,x_test,y_test,clf = model,point_amount =point_amount,scoring = scoring,space = space)
        '============================================================================'
        ## upsampling validation
        fold_test_up = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
        try:
            fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        ##upsampling the validation set          ## set the train set for single fold
        '============================================================================'
        ## set the upsampling validation for single fold
        one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
        zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
        one_message_test_up = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state=42)
        fold_test_up = pd.concat([one_message_test_up,zero_message_test])
        fold_test_up = shuffle(fold_test_up,random_state = random_state)
        fold_test_up = fold_test_up.reset_index(drop = True)
        x_test_up = fold_test_up.drop(columns=["Class"])
        y_test_up = fold_test_up['Class']
        
        '============================================================================'
        '''
        x_fold_test = fold_test.drop(columns=['Class'])
        y_fold_test = fold_test["Class"]
        ros = RandomOverSampler(random_state=42)
        x_test_ros, y_test_ros= ros.fit_resample(x_fold_test, y_fold_test)
        #print(sorted(Counter(y_test_ros).items()))
        #print(x_test_ros.shape,y_test_ros.shape)
        test_up = x_test_ros
        test_up['Class'] = y_test_ros
        test_up = test_up.sample(frac=1,axis =1).reset_index(drop=True)
        x_test_up = test_up.drop(columns=['Class'])
        y_test_up = test_up['Class']
        '''
        '============================================================================'
        ## log the train/validation loss for upsampling
        train_loss_up[i-1],test_loss_up[i-1] = plot_learning_curves(x_train,y_train,x_test_up,y_test_up,clf = model,point_amount=point_amount,scoring = scoring,space=space)

        del (fold_train,fold_test_up)
        if (space=='geomspace'):
            rng = [int(i) for i in np.geomspace(5, x_train.shape[0], point_amount)][1:]
        elif (space =='linspace'):
            rng = [int(i) for i in np.linspace(5, x_train.shape[0], point_amount)][1:]
        
       
    if sub_plot == True:
    ## plot figure 
        plt.figure(2)
        for i in range(len(train_loss)):
            plt.plot(rng, train_loss[i], color='blue', label='Training loss')
            #plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
            plt.plot(rng, test_loss[i], color='orange',linestyle='--', label='Validation loss')
            #plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
            plt.plot(rng, test_loss_up[i], color='red',linestyle='-', label='Validation(up) loss')
            #plt.ylim(0,0.5)
        plt.ylim([-0.05, 1.05])
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model error')
        plt.legend('lower right')
    elif sub_plot == False:
            pass

## compute the average loss for the learning curve
    for i in range(len(train_loss)):
        total_train_loss_pd['{}_fold'.format(i+1)] = train_loss_up[i]
        total_test_loss_pd['{}_fold'.format(i+1)] = test_loss[i]
        total_test_loss_up_pd['{}_fold'.format(i+1)]= test_loss_up[i]
    total_train_loss_pd['average'] = total_train_loss_pd.sum(axis = 1)/cv
    total_train_loss_pd['min'] = total_train_loss_pd.min(axis =1)
    total_train_loss_pd['max'] = total_train_loss_pd.max(axis = 1)
    
    total_test_loss_pd['average'] = total_test_loss_pd.sum(axis = 1)/cv
    total_test_loss_pd['min'] = total_test_loss_pd.min(axis = 1)
    total_test_loss_pd['max'] = total_test_loss_pd.max(axis = 1)

    total_test_loss_up_pd['average'] = total_test_loss_up_pd.sum(axis = 1)/cv
    total_test_loss_up_pd['min'] = total_test_loss_up_pd.min(axis =1)
    total_test_loss_up_pd['max'] = total_test_loss_up_pd.max(axis = 1)
    print(rng)
    if plot == True:
        plt.figure(cv+1)
        plt.plot(rng, list(total_train_loss_pd['average']), color='blue', label='Training loss')
        plt.fill_between(rng, total_train_loss_pd['min'], total_train_loss_pd['max'], alpha=0.15, color='blue')

        plt.plot(rng, list(total_test_loss_pd['average']), color='orange',linestyle='--', label='Validation loss')
        plt.fill_between(rng, total_test_loss_pd['min'], total_test_loss_pd['max'], alpha=0.15, color='orange')

        plt.plot(rng, list(total_test_loss_up_pd['average']), color='red',linestyle='-', label='Validation(up) loss')
        plt.fill_between(rng, total_test_loss_up_pd['min'], total_test_loss_up_pd['max'], alpha=0.15, color='red')
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model loss')
        plt.ylim([0, 1])
        #plt.ylim(0,0.05)
        if scoring == 'f1':
            plt.legend(loc='lower right')
        else:
            plt.legend(loc ='upper right')
    elif plot ==False:
        pass
    plt.show()
    if loss_log ==True:
        return(train_loss,test_loss,train_loss_up,test_loss_up)
    elif loss_log ==False:
        return(
                [
               {'test_average':list(total_test_loss_pd['average'])},{'test_max':list(total_test_loss_pd['max'])},{'test_min':list(total_test_loss_pd['min'])},
               {'test_up_average':list(total_test_loss_up_pd['average'])},{'test_up_max':list(total_test_loss_up_pd['max'])},{'test_up_min':list(total_test_loss_up_pd['min'])},
               {'train_average':list(total_train_loss_pd['average'])},{'train_max':list(total_train_loss_pd['max'])},{'train_min':list(total_train_loss_pd['min'])},
               {'rng':list(rng)}
                ]
               )


def kfold_self_score(
    cv = 5,
    model =None,
    point_amount = 2,
    scoring = ['misclassification error'],
    seperate = False,
    random_state=30,
    path = 'k_fold_train_validation_42',
    space = 'linspace',
    model_name = None,
    ):
    '''
    out_put -- the accuracy or other parameter in confusion matrix for the non upsampling validation set
    cv : inter, default:5 -- input how many fold we want to do
    model:default: None -- input the training model 
    point_amount: inter,default:20 -- set the train_size(input as how many point want to be seperate by max length of training data)
    sub_plot :boolean,default:False  -- True if want to plot the learning curve for each fold
    loss_log: boolean,default:False -- True if want to get the loss log for each fold,
                                        False for the average loss for 
                                        -train
                                        -nonupsampling validation
                                        -upsampling validation
    plot :boolean,default:True --True if want to plot the learning for after average
    scoring : lisg[str] (default: 'misclassification error')
        If not 'misclassification error', accepts the following metrics
        (from scikit-learn):
        {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
        'f1_weighted', 'f1_samples', 'log_loss',
        'precision', 'recall', 'roc_auc',
        'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
        'median_absolute_error', 'r2'}
    seperate:boolean, if we want to save the score for each fold 
    path: the folder for the datas
    '''
    specificity=[]
    score = pd.DataFrame(columns=[])
    for i in range(1,cv+1):
        score['{}'.format(i)] = []
        specificity.append([])
    for type in scoring:
        print('start:',type)
        confuse =[]
        save = []
        for i in range(1,cv+1):
            confuse.append([])
            
        for i in trange(1,cv+1):
            fold_train = pd.read_csv(path+'/{}-fold_train_z.csv'.format(i))
            fold_test = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
            ##rename(消去columns項內非英文以及數字的解)
            try:
                fold_train = fold_train.drop(columns=['Unnamed: 0'])
                fold_test = fold_test.drop(columns=['Unnamed: 0'])
            except:
                pass
            fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

            '==============================================================================='
            '''
            x_fold_train = fold_train.drop(columns=['Class'])
            y_fold_train = fold_train["Class"]
            ros = RandomOverSampler(random_state=42)
            x_train_ros, y_train_ros= ros.fit_resample(x_fold_train, y_fold_train)
            #print(sorted(Counter(y_train_ros).items()))
            #print(x_train_ros.shape,y_train_ros.shape)
            train_up = x_train_ros
            train_up['Class'] = y_train_ros
            train_up = train_up.sample(frac=1,axis =1).reset_index(drop=True)
            #print(train_up)
            # set the train set for single fold
            x_train = train_up.drop(columns=['Class'])
            y_train = train_up['Class']
            '''
            '==============================================================================='
            
            ## upsampling train set 
            one_message = fold_train[fold_train["Class"] == 1 ]
            zero_message =fold_train[fold_train["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=42)
            fold_train = pd.concat([one_message_up,zero_message])
            fold_train = shuffle(fold_train,random_state = random_state)
            fold_train= fold_train.reset_index(drop = True)
            ## set the train set for single fold
            x_train = fold_train.drop(columns=['Class'])
            y_train = fold_train['Class']
            
            '=================================================================================='
            ## set the non upsampling validation for the single fold
            fold_test = shuffle(fold_test,random_state = random_state)
            fold_test= fold_test.reset_index(drop = True)
            x_test = fold_test.drop(columns=["Class"])
            y_test = fold_test['Class']
            if type != 'specificity':
            ## log train/validationS loss for non upsampling
                confuse[i-1] = plot_learning_curves(x_train,y_train,x_test,y_test,clf = model,point_amount =point_amount,scoring=type,space = space)#單個fold會回傳兩個list,前面那個是train set 得到的score,後面為validation得到
                #print(confuse)
            ## get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
            elif type =='specificity':
                spe_model = model.fit(x_train,y_train)
                y_pred = spe_model.predict(x_test)
                if y_pred.ndim != 1:
                    y_pred = y_pred.reshape(y_pred.shape[0],)
                ## log train/validationS loss for non upsampling
                ## tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                # specificity[i-1] = (tn / (tn+fp))
                specificity[i-1] = recall_score(y_test, y_pred, pos_label=0)
            del (fold_train,fold_test)

        if type !='specificity':
            for num in range(0,len(confuse)):
                save.append(confuse[num][1][point_amount-2])
            score.loc['{}'.format(type)] = save
        elif type == 'specificity':
            for num in range(0,len(specificity)):
                save.append(specificity[num])
            score.loc['{}'.format(type)] = save
        print('finished:',type)
        
    score['average'] = score.sum(axis = 1)/cv
    if seperate == True:
        return(save)
    else:
        return(score)


def kfold_self_score_up(
    cv = 5,
    model =None,
    point_amount = 2,
    scoring = ['misclassification error'],
    seperate = False,
    path = 'k_fold_train_validation_42',
    space = 'linspace',
    random_state = 30,
    model_name = None
    ):
    '''
    out_put -- the accuracy or other parameter in confusion matrix for the upsampling validation set
    cv : inter, default:5 -- input how many fold we want to do
    model:default: None -- input the training model 
    point_amount: inter,default:20 -- set the train_size(input as how many point want to be seperate by max length of training data)
    sub_plot :boolean,default:False  -- True if want to plot the learning curve for each fold
    loss_log: boolean,default:False -- True if want to get the loss log for each fold,
                                        False for the average loss for 
                                        -train
                                        -nonupsampling validation
                                        -upsampling validation
    plot :boolean,default:True --True if want to plot the learning for after average
    scoring : lisg[str] (default: 'misclassification error')
        If not 'misclassification error', accepts the following metrics
        (from scikit-learn):
        {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
        'f1_weighted', 'f1_samples', 'log_loss',
        'precision', 'recall', 'roc_auc',
        'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
        'median_absolute_error', 'r2'}
    seperate:boolean, if we want to save the score for each fold 
    path: the folder for the datas
    '''
    specificity=[]
    score = pd.DataFrame(columns=[])
    for i in range(1,cv+1):
        score['{}'.format(i)] = []
        specificity.append([])
    for type in scoring:
        print('start:',type)
        confuse =[]
        save = []
        
        for i in range(1,cv+1):
            confuse.append([])
        for i in trange(1,cv+1):
            fold_train = pd.read_csv(path+'/{}-fold_train_z.csv'.format(i))
            fold_test = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
            #rename(消去columns項內非英文以及數字的解)
            try:
                fold_train = fold_train.drop(columns=['Unnamed: 0'])
                fold_test = fold_test.drop(columns=['Unnamed: 0'])
            except:
                pass
            fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            '============================================================================='
            '''
            x_fold_train = fold_train.drop(columns=['Class'])
            y_fold_train = fold_train["Class"]
            ros = RandomOverSampler(random_state=42)
            x_train_ros, y_train_ros= ros.fit_resample(x_fold_train, y_fold_train)
            #print(sorted(Counter(y_train_ros).items()))
            #print(x_train_ros.shape,y_train_ros.shape)
            train_up = x_train_ros
            train_up['Class'] = y_train_ros
            train_up = train_up.sample(frac=1,axis =1).reset_index(drop=True)
            #print(train_up)
            # set the train set for single fold
            x_train = train_up.drop(columns=['Class'])
            y_train = train_up['Class']
            '''
            '=============================================================================='
            ## upsampling train set 
            one_message = fold_train[fold_train["Class"] == 1 ]
            zero_message =fold_train[fold_train["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=42)
            fold_train = pd.concat([one_message_up,zero_message])
            fold_train = shuffle(fold_train,random_state = random_state)
            fold_train= fold_train.reset_index(drop = True)
            # set the train set for single fold
            x_train = fold_train.drop(columns=['Class'])
            y_train = fold_train['Class']
            #print('x_train:',x_train,'y_train:',y_train)
            '========================================================================='
            '''
            x_fold_test = fold_test.drop(columns=['Class'])
            y_fold_test = fold_test["Class"]
            ros = RandomOverSampler(random_state=42)
            x_test_ros, y_test_ros= ros.fit_resample(x_fold_test, y_fold_test)
            #print(sorted(Counter(y_test_ros).items()))
            #print(x_test_ros.shape,y_test_ros.shape)
            test_up = x_test_ros
            test_up['Class'] = y_test_ros
            test_up = test_up.sample(frac=1,axis =0).reset_index(drop=True)
            #print(train_up)
            # set the train set for single fold
            x_test = test_up.drop(columns=['Class'])
            y_test = test_up['Class']
            '''
            '========================================================================='
            ## set the non upsampling validation for the single fold
            one_message_test = fold_test[fold_test["Class"] == 1 ]
            zero_message_test =fold_test[fold_test["Class"] == 0]
            one_message_up_test = resample(one_message_test,
                        replace=True,
                        n_samples=len(zero_message_test),
                        random_state=42)
            fold_test = pd.concat([one_message_up_test,zero_message_test])
            fold_test = shuffle(fold_test,random_state = random_state)
            fold_test= fold_test.reset_index(drop = True)         
            x_test = fold_test.drop(columns=["Class"])
            y_test = fold_test['Class']
            '============================================================='
            if type != 'specificity':
            ## log train/validationS loss for non upsampling
                confuse[i-1] = plot_learning_curves(x_train,y_train,x_test,y_test,clf = model,point_amount =point_amount,scoring=type,space=space)#單個fold會回傳兩個list,前面那個是train set 得到的score,後面為validation得到
            ## get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
            elif type =='specificity':
                spe_model = model.fit(x_train,y_train)
                y_pred = model.predict(x_test)
                if y_pred.ndim != 1:
                    y_pred = y_pred.reshape(y_pred.shape[0],)    
                ## log train/validationS loss for non upsampling
                ##tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                #specificity[i-1] = (tn / (tn+fp))
                specificity[i-1] = recall_score(y_test, y_pred, pos_label=0)
            del (fold_train,fold_test)

        if type !='specificity':
            for num in range(0,len(confuse)):
                save.append(confuse[num][1][point_amount-2])
            score.loc['{}'.format(type)] = save
        elif type == 'specificity':
            for num in range(0,len(specificity)):
                save.append(specificity[num])
            score.loc['{}'.format(type)] = save
        del(save,confuse)
        print('finished:',type)
    score['average'] = score.sum(axis = 1)/cv
    if seperate == True:
        return(save)
    else:
        return(score)


def roc_plot(
    model = None,
    fold = 5,
    heat_map = False,
    path = 'k_fold_train_validation_42',
    path_2='k_fold_train_validation_42',
    random_state_1=30,
    random_state_2=30,
    model_amount = 1,
    model_2 = None,
    color_origin='orange',
    color_up = 'orange',
    color_origin_2='green',
    color_up_2='green'
    ):
    '''
    out_put : the roc plot for the single fold
    cv : inter, default:5 -- input how many fold we want to do
    model:default:None  -- input the training model 
    heat_map: boolean,default:False -- True if we want to output the heat map 
    random_state_1 : the random seed for the shuffle and oversampling for model1
    random_state_2: the random seed for the shuffle and oversampling for model2
    path: the folder for the datas that we train model 1
    path_2: the folder for the datas that we train model 2
    model_amount : inter, 1 or 2
    color_origin: only modifiy the prc color for second model(origin distribution)
    color_up: only modeifiy the prc color for second model(upsampling distribution)    
    '''
    fper = []
    tper = []
    fper_up = []
    tper_up = []
    auc = []
    auc_up =[]
    fper_2 = []
    tper_2 = []
    fper_up_2 = []
    tper_up_2 = []
    auc_2 = []
    auc_up_2 =[]
    if model_amount==1:
        for i in trange(1,fold+1):   
            #print('fold is :',i)   
            '==============================upsampling train/origin======================='
            fold_train = pd.read_csv(path+'/{}-fold_train_z.csv'.format(i))
            fold_test = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
            ## rename(消去columns項內非英文以及數字的解)
            try:
                fold_train = fold_train.drop(columns=['Unnamed: 0'])
                fold_test = fold_test.drop(columns=['Unnamed: 0'])
            except:
                pass
            fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            ## upsampling train set 
            one_message = fold_train[fold_train["Class"] == 1 ]
            zero_message =fold_train[fold_train["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=42)
            fold_train = pd.concat([one_message_up,zero_message])
            fold_train = shuffle(fold_train,random_state = random_state_1)
            fold_train= fold_train.reset_index(drop = True)
            ## set the train set for single fold
            x_train = fold_train.drop(columns=['Class'])
            y_train = fold_train['Class']
            ## set the non upsampling validation for the single fold
            fold_test = shuffle(fold_test,random_state = random_state_1)
            fold_test= fold_test.reset_index(drop = True)
            x_test = fold_test.drop(columns=["Class"])
            y_test = fold_test['Class']
            ## get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
            roc_model = model.fit(x_train,y_train)
            y_pred = roc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖,取為1的機率
            if y_pred.ndim != 1:
                y_pred = y_pred.reshape(y_pred.shape[0],)

            y_pred_hm = roc_model.predict(x_test)
            ## log train/validationS loss for non upsampling
            fper_i , tper_i , threshold = roc_curve(y_test,y_pred, pos_label=1)
            fper.append(fper_i)
            tper.append(tper_i)
            '==================================upsampling train/upsampling test======================================='
            fold_test_up = pd.read_csv(path+'/{}-fold_test_z.csv'.format(fold))
            try:
                fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
            except:
                pass
            fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
            zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
            #unsample test set 
            one_message_test_up = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state=42)
            fold_test_up = pd.concat([one_message_test_up,zero_message_test])
            fold_test_up = shuffle(fold_test_up,random_state = random_state_1)
            fold_test_up= fold_test_up.reset_index(drop = True)
            x_test_up = fold_test_up.drop(columns = ['Class'])
            y_test_up = fold_test_up["Class"]
            '=======================================auc==========================================================='
            y_pred_up = roc_model.predict_proba(x_test_up)[:,1]
            if y_pred_up.ndim != 1:
                y_pred_up = y_pred_up.reshape(y_pred_up.shape[0],)

            y_pred_up_hm = roc_model.predict(x_test_up)
            fper_up_i,tper_up_i,threshold_up = roc_curve(y_test_up,y_pred_up, pos_label=1)
            fper_up.append(fper_up_i)
            tper_up.append(tper_up_i)
            '=======================================auc==========================================================='
            auc.append(roc_auc_score(y_test,y_pred))
            auc_up.append(roc_auc_score(y_test_up,y_pred_up))
            if heat_map == True:
                cnf_matrix = confusion_matrix(y_test, y_pred_hm)
                plt.figure(2)
                plot_confusion_matrix(cnf_matrix)
                plt.figure(3)
                cnf_matrix2 = confusion_matrix(y_test_up, y_pred_up_hm)
                plot_confusion_matrix(cnf_matrix2)
            else:
                pass
        del (fold_train,fold_test)
        plt.figure(1)
        fper_max,tper_max,fper_min,tper_min,fper_up_max,tper_up_max,fper_up_min,tper_up_min,auc_avg,auc_up_avg =plot_roc_curve(fper,tper,fper_up,tper_up,auc,auc_up,color_origin,color_up)
        print('auc = ',auc,'auc_up =',auc_up)        
        plt.show()
        roc_data = {
            'fper_max':list(fper_max),
            'tper_max':list(tper_max),
            'fper_min':list(fper_min),
            'tper_min':list(tper_min),
            'fper_up_max':list(fper_up_max),
            'tper_up_max':list(tper_up_max),
            'fper_up_min':list(fper_up_min),
            "tper_up_min":list(tper_up_min),
            "auc_avg":auc_avg,
            "auc_up_avg":auc_up_avg,
        }
        return(roc_data)

    elif model_amount ==2:    
        for i in range(1,fold+1):      
            '==============================upsampling train/origin model 1======================='
            fold_train = pd.read_csv(path+'/{}-fold_train_z.csv'.format(i))
            fold_test = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
            #rename(消去columns項內非英文以及數字的解)
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
            fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            #print(i,'-1')
            # upsampling train set 
            one_message = fold_train[fold_train["Class"] == 1 ]
            zero_message =fold_train[fold_train["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=random_state_1)
            fold_train = pd.concat([one_message_up,zero_message])
            fold_train = shuffle(fold_train,random_state = random_state_1)
            fold_train= fold_train.reset_index(drop = True)
            # set the train set for single fold
            x_train = fold_train.drop(columns=['Class'])
            y_train = fold_train['Class']
            #print('x_train:',x_train,'y_train:',y_train)
            # set the non upsampling validation for the single fold
            fold_test = shuffle(fold_test,random_state = random_state_1)
            fold_test= fold_test.reset_index(drop = True)
            x_test = fold_test.drop(columns=["Class"])
            y_test = fold_test['Class']
            '==============================upsampling train/origin model 2======================='
            fold_train_2 = pd.read_csv(path_2+'/{}-fold_train_z.csv'.format(i))
            fold_test_2 = pd.read_csv(path_2+'/{}-fold_test_z.csv'.format(i))
            #rename(消去columns項內非英文以及數字的解)
            fold_train_2 = fold_train_2.drop(columns=['Unnamed: 0'])
            fold_test_2 = fold_test_2.drop(columns=['Unnamed: 0'])
            fold_train_2 = fold_train_2.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test_2 = fold_test_2.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            #print(i,'-1')
            # upsampling train set 
            one_message = fold_train_2[fold_train_2["Class"] == 1 ]
            zero_message =fold_train_2[fold_train_2["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=random_state_2)
            fold_train_2 = pd.concat([one_message_up,zero_message])
            fold_train_2 = shuffle(fold_train_2,random_state = random_state_2)
            fold_train_2= fold_train_2.reset_index(drop = True)
            # set the train set for single fold
            x_train_2 = fold_train_2.drop(columns=['Class'])
            y_train_2 = fold_train_2['Class']
            #print('x_train:',x_train,'y_train:',y_train)
            # set the non upsampling validation for the single fold
            fold_test_2 = shuffle(fold_test_2,random_state = random_state_2)
            fold_test_2= fold_test_2.reset_index(drop = True)
            x_test_2 = fold_test_2.drop(columns=["Class"])
            y_test_2 = fold_test_2['Class']
            '===========================training model==========================================================='
            #model 1
            # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
            roc_model = model.fit(x_train,y_train)
            y_pred = roc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
            y_pred_hm = roc_model.predict(x_test)
            #log train/validationS loss for non upsampling
            fper_i , tper_i , threshold = roc_curve(y_test,y_pred, pos_label=1)
            fper.append(fper_i)
            tper.append(tper_i)
            ##model 2
            roc_model_2 = model_2.fit(x_train_2,y_train_2)
            y_pred_2 = roc_model_2.predict_proba(x_test_2)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
            y_pred_hm_2 = roc_model_2.predict(x_test_2)
            #log train/validationS loss for non upsampling
            fper_i_2 , tper_i_2 , threshold = roc_curve(y_test_2,y_pred_2, pos_label=1)
            fper_2.append(fper_i_2)
            tper_2.append(tper_i_2)
            '==================================upsampling train/upsampling test model 1======================================='
            fold_test_up = pd.read_csv(path+'/{}-fold_test_z.csv'.format(fold))
            fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
            fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
            zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
            one_message_test_up = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state=random_state_1)
            fold_test_up = pd.concat([one_message_test_up,zero_message_test])
            fold_test_up = shuffle(fold_test_up,random_state = random_state_1)
            fold_test_up= fold_test_up.reset_index(drop = True)
            x_test_up = fold_test_up.drop(columns = ['Class'])
            y_test_up = fold_test_up["Class"]
            '==================================upsampling train/upsampling test model 2======================================='
            fold_test_up_2 = pd.read_csv(path_2+'/{}-fold_test_z.csv'.format(fold))
            fold_test_up_2 = fold_test_up_2.drop(columns=['Unnamed: 0'])
            fold_test_up_2 = fold_test_up_2.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            one_message_test = fold_test_up_2[fold_test_up_2["Class"] == 1 ]
            zero_message_test =fold_test_up_2[fold_test_up_2["Class"] == 0]
            one_message_test_up = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state=random_state_2)
            fold_test_up_2 = pd.concat([one_message_test_up,zero_message_test])
            fold_test_up_2 = shuffle(fold_test_up_2,random_state = random_state_2)
            fold_test_up_2= fold_test_up_2.reset_index(drop = True)
            x_test_up_2 = fold_test_up_2.drop(columns = ['Class'])
            y_test_up_2 = fold_test_up_2["Class"]
            '=======================================auc==========================================================='
            #modle 1
            y_pred_up = roc_model.predict_proba(x_test_up)[:,1]
            y_pred_up_hm = roc_model.predict(x_test_up)
            fper_up_i,tper_up_i,threshold_up = roc_curve(y_test_up,y_pred_up, pos_label=1)
            fper_up.append(fper_up_i)
            tper_up.append(tper_up_i)
            ## model 2 
            y_pred_up_2 = roc_model_2.predict_proba(x_test_up_2)[:,1]
            y_pred_up_hm_2 = roc_model_2.predict(x_test_up_2)
            fper_up_i_2,tper_up_i_2,threshold_up = roc_curve(y_test_up_2,y_pred_up_2, pos_label=1)
            fper_up_2.append(fper_up_i_2)
            tper_up_2.append(tper_up_i_2)
            '=======================================auc==========================================================='
            auc.append(roc_auc_score(y_test,y_pred))
            auc_up.append(roc_auc_score(y_test_up,y_pred_up))
            ##for model two 
            auc_2.append(roc_auc_score(y_test,y_pred_2))
            auc_up_2.append(roc_auc_score(y_test_up,y_pred_up_2))
            if heat_map == True:
                cnf_matrix = confusion_matrix(y_test, y_pred_hm)
                plt.figure(2)
                plot_confusion_matrix(cnf_matrix)
                plt.figure(3)
                cnf_matrix2 = confusion_matrix(y_test_up, y_pred_up_hm)
                plot_confusion_matrix(cnf_matrix2)
            else:
                pass
        del (fold_train,fold_test)
        plt.figure(1)
        plot_roc_curve(fper,tper,fper_up,tper_up,auc,auc_up,color_origin,color_up)
        plot_roc_curve(fper_2,tper_2,fper_up_2,tper_up_2,auc_2,auc_up_2,color_origin_2,color_up_2)
        print('auc = ',auc,'auc_up =',auc_up)
        plt.show()


def prc_plot(
    model =None,
    model_2 = None,
    fold = 5,
    heat_map = False,
    path = 'k_fold_train_validation_42',
    path_2=None,
    model_amount = 1,
    color_origin='orange',
    color_up = 'orange',
    color_origin_2='green',
    color_up_2='green',
    random_state_1 = 30,
    random_state_2 = 30,
    model_save_path = None,
    ):
    '''
    out_put : the roc plot for the single fold
    cv : inter, default:5 -- input how many fold we want to do
    model:default:None  -- input the training model 
    heat_map: boolean,default:False -- True if we want to output the heat map 
    random_state_1 : the random seed for the shuffle and oversampling for model1
    random_state_2: the random seed for the shuffle and oversampling for model2
    path: the folder for the datas that we train model 1
    path_2: the folder for the datas that we train model 2
    model_amount : inter, 1 or 2
    color_origin: only modifiy the prc color for second model(origin distribution)
    color_up: only modeifiy the prc color for second model(upsampling distribution)
    '''
    auc_list = []
    auc_list_up = []
    precision = []
    recall = []
    precision_up = []
    recall_up = []
    auc_list_2 = []
    auc_list_up_2 = []
    precision_2 = []
    recall_2 = []
    precision_up_2 = []
    recall_up_2 = []


    if model_amount == 1:
        for i in trange(1,fold+1):
            #print('fold is :',i)
            '==============================upsampling train/origin model_1======================='
            fold_train = pd.read_csv(path+'/{}-fold_train_z.csv'.format(i))
            fold_test = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
            #rename(消去columns項內非英文以及數字的解)
            try:
                fold_train = fold_train.drop(columns=['Unnamed: 0'])
                fold_test = fold_test.drop(columns=['Unnamed: 0'])
            except:
                pass
            fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            #print(i,'-1')
            # upsampling train set 
            one_message = fold_train[fold_train["Class"] == 1 ]
            zero_message =fold_train[fold_train["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=42)
            fold_train = pd.concat([one_message_up,zero_message])
            fold_train = shuffle(fold_train,random_state = random_state_1)
            fold_train= fold_train.reset_index(drop = True)
            # set the train set for single fold
            x_train = fold_train.drop(columns=['Class'])
            y_train = fold_train['Class']
            #print('x_train:',x_train,'y_train:',y_train)
            # set the non upsampling validation for the single fold
            fold_test = shuffle(fold_test,random_state = random_state_1)
            fold_test= fold_test.reset_index(drop = True)
            x_test = fold_test.drop(columns=["Class"])
            y_test = fold_test['Class']

            '=============================model training=============================================================='
            # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
            prc_model = model.fit(x_train,y_train)
            joblib.dump(prc_model, model_save_path+'/{}-model'.format(i))
            y_pred = prc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，取為1的機率
            if y_pred.ndim != 1:
                y_pred = y_pred.reshape(y_pred.shape[0],)
            y_pred_hm = prc_model.predict(x_test)
            #log train/validationS loss for non upsampling
            precision_i , recall_i , threshold = precision_recall_curve(y_test,y_pred, pos_label=1)
            precision.append(precision_i)
            recall.append(recall_i)
            '==================================upsampling train/upsampling test======================================='
            fold_test_up = pd.read_csv(path+'/{}-fold_test_z.csv'.format(fold))
            try:
                fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
            except:
                pass
            fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
            zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
            one_message_test_up = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state=42)
            fold_test_up = pd.concat([one_message_test_up,zero_message_test])
            fold_test_up = shuffle(fold_test_up,random_state = 30)
            fold_test_up= fold_test_up.reset_index(drop = True)
            x_test_up = fold_test_up.drop(columns = ['Class'])
            y_test_up = fold_test_up["Class"]
            '=======================================auc==========================================================='
            y_pred_up = prc_model.predict_proba(x_test_up)[:,1]
            if y_pred_up.ndim != 1:
                y_pred_up = y_pred_up.reshape(y_pred_up.shape[0],)            
            y_pred_up_hm = prc_model.predict(x_test_up)
            precision_up_i,recall_up_i,threshold_up = precision_recall_curve(y_test_up,y_pred_up, pos_label=1)
            precision_up.append(precision_up_i)
            recall_up.append(recall_up_i)
            '=======================================auc==========================================================='
            auc_list.append(auc(recall[i-1],precision[i-1]))
            auc_list_up.append(auc(recall_up[i-1],precision_up[i-1]))
            del (fold_train,fold_test)
        plt.figure(1)
        recall_min,precision_min,recall_max,precision_max,recall_min_up,precision_min_up,recall_max_up,precision_max_up,auc_avg,auc_up_avg = plot_prc_curve(precision,recall,precision_up,recall_up,auc_list,auc_list_up)
    else:
        for i in range(1,fold+1):
            '==============================upsampling train/origin model_1======================='
            fold_train = pd.read_csv(path+'/{}-fold_train_z.csv'.format(i))
            fold_test = pd.read_csv(path+'/{}-fold_test_z.csv'.format(i))
            fold_train_2 = pd.read_csv(path_2+'/{}-fold_train_z.csv'.format(i))
            fold_test_2 = pd.read_csv(path_2+'/{}-fold_test_z.csv'.format(i))
            #rename(消去columns項內非英文以及數字的解)
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
            fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            #print(i,'-1')
            # upsampling train set 
            one_message = fold_train[fold_train["Class"] == 1 ]
            zero_message =fold_train[fold_train["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=random_state_1)
            fold_train = pd.concat([one_message_up,zero_message])
            fold_train = shuffle(fold_train,random_state = random_state_1)
            fold_train= fold_train.reset_index(drop = True)
            # set the train set for single fold
            x_train = fold_train.drop(columns=['Class'])
            y_train = fold_train['Class']
            #print('x_train:',x_train,'y_train:',y_train)
            # set the non upsampling validation for the single fold
            fold_test = shuffle(fold_test,random_state = random_state_1)
            fold_test= fold_test.reset_index(drop = True)
            x_test = fold_test.drop(columns=["Class"])
            y_test = fold_test['Class']
            '==============================upsampling train/origin model_2======================='
            fold_train_2 = pd.read_csv(path_2+'/{}-fold_train_z.csv'.format(i))
            fold_test_2 = pd.read_csv(path_2+'/{}-fold_test_z.csv'.format(i))
            #rename(消去columns項內非英文以及數字的解)
            fold_train_2 = fold_train_2.drop(columns=['Unnamed: 0'])
            fold_test_2 = fold_test_2.drop(columns=['Unnamed: 0'])
            fold_train_2 = fold_train_2.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            fold_test_2 = fold_test_2.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            #print(i,'-1')
            # upsampling train set 
            one_message = fold_train_2[fold_train_2["Class"] == 1 ]
            zero_message =fold_train_2[fold_train_2["Class"] == 0]
            one_message_up = resample(one_message,
                        replace=True,
                        n_samples=len(zero_message),
                        random_state=random_state_2)
            fold_train_2 = pd.concat([one_message_up,zero_message])
            fold_train_2= shuffle(fold_train_2,random_state = random_state_2)
            fold_train_2= fold_train_2.reset_index(drop = True)
            # set the train set for single fold
            x_train_2 = fold_train_2.drop(columns=['Class'])
            y_train_2 = fold_train_2['Class']
            #print('x_train:',x_train,'y_train:',y_train)
            # set the non upsampling validation for the single fold
            fold_test_2 = shuffle(fold_test_2,random_state = random_state_1)
            fold_test_2= fold_test_2.reset_index(drop = True)
            x_test_2 = fold_test_2.drop(columns=["Class"])
            y_test_2 = fold_test_2['Class']
            '=============================model training=============================================================='
            #model 1
            # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
            prc_model = model.fit(x_train,y_train)
            y_pred = prc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
            y_pred_hm = prc_model.predict(x_test)
            #log train/validationS loss for non upsampling
            precision_i , recall_i , threshold = precision_recall_curve(y_test,y_pred, pos_label=1)
            precision.append(precision_i)
            recall.append(recall_i)
            # model 2
            prc_model_2 = model_2.fit(x_train_2,y_train_2)
            y_pred_2 = prc_model_2.predict_proba(x_test_2)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
            y_pred_hm_2 = prc_model_2.predict(x_test_2)
            #log train/validationS loss for non upsampling
            precision_i_2 , recall_i_2 , threshold_2 = precision_recall_curve(y_test_2,y_pred_2, pos_label=1)
            precision_2.append(precision_i_2)
            recall_2.append(recall_i_2)
            '==================================upsampling train/upsampling test model_1======================================='
            fold_test_up = pd.read_csv(path+'/{}-fold_test_z.csv'.format(fold))
            fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
            fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
            zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
            one_message_test_up = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state= random_state_1)
            fold_test_up = pd.concat([one_message_test_up,zero_message_test])
            fold_test_up = shuffle(fold_test_up,random_state = random_state_1)
            fold_test_up= fold_test_up.reset_index(drop = True)
            x_test_up = fold_test_up.drop(columns = ['Class'])
            y_test_up = fold_test_up["Class"]
            '==================================upsampling train/upsampling test model_1======================================='
            fold_test_up_2 = pd.read_csv(path_2+'/{}-fold_test_z.csv'.format(fold))
            fold_test_up_2 = fold_test_up_2.drop(columns=['Unnamed: 0'])
            fold_test_up_2 = fold_test_up_2.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            one_message_test = fold_test_up_2[fold_test_up_2["Class"] == 1 ]
            zero_message_test =fold_test_up_2[fold_test_up_2["Class"] == 0]
            one_message_test_up = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state= random_state_2)
            fold_test_up_2 = pd.concat([one_message_test_up,zero_message_test])
            fold_test_up_2 = shuffle(fold_test_up_2,random_state = random_state_2)
            fold_test_up_2= fold_test_up_2.reset_index(drop = True)
            x_test_up_2 = fold_test_up_2.drop(columns = ['Class'])
            y_test_up_2 = fold_test_up_2["Class"]
            '=======================================auc==========================================================='
            #model 1
            y_pred_up = prc_model.predict_proba(x_test_up)[:,1]
            y_pred_up_hm = prc_model.predict(x_test_up)
            precision_up_i,recall_up_i,threshold_up = precision_recall_curve(y_test_up,y_pred_up, pos_label=1)
            precision_up.append(precision_up_i)
            recall_up.append(recall_up_i)
            #model 2
            y_pred_up_2 = prc_model_2.predict_proba(x_test_up_2)[:,1]
            y_pred_up_hm_2 = prc_model_2.predict(x_test_up_2)
            precision_up_i_2,recall_up_i_2,threshold_up_2 = precision_recall_curve(y_test_up_2,y_pred_up_2, pos_label=1)
            precision_up_2.append(precision_up_i_2)
            recall_up_2.append(recall_up_i_2)
            '=======================================auc==========================================================='
            auc_list.append(auc(recall[i-1],precision[i-1]))
            auc_list_up.append(auc(recall_up[i-1],precision_up[i-1]))
            auc_list_2.append(auc(recall_2[i-1],precision_2[i-1]))
            auc_list_up_2.append(auc(recall_up_2[i-1],precision_up_2[i-1]))
            del (fold_train,fold_test)
        plt.figure(1)
        plot_prc_curve(precision,recall,precision_up,recall_up,auc_list,auc_list_up,color_origin,color_up)   
        plot_prc_curve(precision_2,recall_2,precision_up_2,recall_up_2,auc_list_2,auc_list_up_2,color_origin_2,color_up_2)    


    #print(len(fper_list[0]),'\n\n',len(tper_list[0]))
    if heat_map == True:
        cnf_matrix = confusion_matrix(y_test, y_pred_hm)
        plt.figure(2)
        plot_confusion_matrix(cnf_matrix)
        plt.figure(3)
        cnf_matrix2 = confusion_matrix(y_test_up, y_pred_up_hm)
        plot_confusion_matrix(cnf_matrix2)
    else:
        pass
    plt.show()
    prc_curve_data = {
                      'recall_min':recall_min,
                      'precision_min':precision_min,
                      'recall_max':recall_max,
                      'precision_max':precision_max,
                      'recall_min_up':recall_min_up,
                      'precision_min_up':precision_min_up,
                      'recall_max_up':recall_max_up,
                      'precision_max_up':precision_max_up,
                      'auc_avg':auc_avg,
                      'auc_up_avg':auc_up_avg
                      }
                      
    return(prc_curve_data)


'''=============functions for testing datas (after tuning hyper parameters)========='''
def test(
    model = None,
    point_amount = 20,
    sub_plot = False,
    loss_log = False,
    plot =True,
    random_state  = 40,
    scoring = "misclassification error",
    file_train = 'k_fold_train_validation_42',
    file_test=None,
    space = 'geomspace'
    ):
    rng =0
    total_train_loss_pd = pd.DataFrame()
    total_test_loss_pd = pd.DataFrame()
    total_test_loss_up_pd = pd.DataFrame()
    fold_train = pd.read_csv(file_train)
    fold_test = pd.read_csv(file_test)
    ## rename(消去columns項內非英文以及數字的解)
    try:
        fold_train = fold_train.drop(columns=['Unnamed: 0'])
        fold_test = fold_test.drop(columns=['Unnamed: 0'])
    except:
        pass
    fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    '==============================================================================='   
    ## upsampling train set 
    one_message = fold_train[fold_train["Class"] == 1 ]
    zero_message =fold_train[fold_train["Class"] == 0]
    one_message_up = resample(one_message,
                replace=True,
                n_samples=len(zero_message),
                random_state=42)
    fold_train = pd.concat([one_message_up,zero_message])
    #fold_train = fold_train.sample(frac=1,axis = 0).reset_index(drop=True)
    fold_train = shuffle(fold_train,random_state = random_state)
    fold_train= fold_train.reset_index(drop = True)
    
    ## set the train set for single fold
    x_train = fold_train.drop(columns=['Class'])
    y_train = fold_train['Class']
    
    ## set the non upsampling validation for the single fold       
    fold_test = shuffle(fold_test,random_state = random_state)
    fold_test = fold_test.reset_index(drop = True)
    x_test = fold_test.drop(columns=["Class"])
    y_test = fold_test['Class']
    

    ## log train/validationS loss for non upsampling
    train_loss,test_loss = plot_learning_curves(x_train,y_train,x_test,y_test,clf = model,point_amount =point_amount,scoring = scoring,space = space)
    '============================================================================'
    ## upsampling validation
    fold_test_up = pd.read_csv(file_test)
    try:
        fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
    except:
        pass
    fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    #upsampling the validation set          # set the train set for single fold
    '============================================================================'
    # set the upsampling validation for single fold
    one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
    zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
    one_message_test_up = resample(one_message_test,
                replace=True,
                n_samples=len(zero_message_test),
                random_state=42)
    fold_test_up = pd.concat([one_message_test_up,zero_message_test])
    fold_test_up = shuffle(fold_test_up,random_state = random_state)
    fold_test_up = fold_test_up.reset_index(drop = True)
    x_test_up = fold_test_up.drop(columns=["Class"])
    y_test_up = fold_test_up['Class']
    
    '============================================================================'
    # log the train/validation loss for upsampling
    #print('x_test_up:',x_test_up,'y_test_up:',y_test_up)
    train_loss_up,test_loss_up = plot_learning_curves(x_train,y_train,x_test_up,y_test_up,clf = model,point_amount=point_amount,scoring = scoring,space=space)

    del (fold_train,fold_test_up)
    if (space=='geomspace'):
        rng = [int(i) for i in np.geomspace(5, x_train.shape[0], point_amount)][1:]
    elif (space =='linspace'):
        rng = [int(i) for i in np.linspace(5, x_train.shape[0], point_amount)][1:]
    if plot == True:
        plt.figure(1)
        plt.plot(rng, list(total_train_loss_pd), color='blue', label='Training loss')
        plt.plot(rng, list(total_test_loss_pd), color='orange',linestyle='--', label='test loss')
        plt.plot(rng, list(total_test_loss_up_pd), color='red',linestyle='-', label='test_up loss')
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model loss')
        #plt.ylim(0,0.05)
        plt.legend(loc='upper right')
    elif plot ==False:
        pass
    plt.show()    


def test_prc(
    model =None,
    model_2 = None,
    random_state_1 = 42,
    random_state_2 = 42,
    random_state_test = 42,
    model_amount = 1,
    color_origin='orange',
    color_up = 'orange',
    color_origin_2='green',
    color_up_2='green',
    file_train = None,
    file_test = None,
    file_train_2 = None
    ):
    '''
    this code not been use any more
    '''
    if model_amount ==1:
        '==============================upsampling train/origin======================='
        fold_train = pd.read_csv(file_train)
        fold_test = pd.read_csv(file_test)
        ## rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        ## upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        fold_train = shuffle(fold_train,random_state = 42)
        fold_train= fold_train.reset_index(drop = True)
        ## set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
        ## set the non upsampling validation for the single fold
        fold_test = shuffle(fold_test,random_state = 42)
        fold_test= fold_test.reset_index(drop = True)
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']
        '=============================model training=============================================================='
        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        prc_model = model.fit(x_train,y_train)
        y_pred = prc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
        y_pred_hm = prc_model.predict(x_test)
        #log train/validationS loss for non upsampling
        precision_i , recall_i , threshold = precision_recall_curve(y_test,y_pred, pos_label=1)
        '==================================upsampling train/upsampling test======================================='
        fold_test_up = pd.read_csv(file_test)
        try:
            fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
        zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
        one_message_test_up = resample(one_message_test,
                replace=True,
                n_samples=len(zero_message_test),
                random_state=42)
        fold_test_up = pd.concat([one_message_test_up,zero_message_test])
        fold_test_up = shuffle(fold_test_up,random_state = 42)
        fold_test_up= fold_test_up.reset_index(drop = True)
        x_test_up = fold_test_up.drop(columns = ['Class'])
        y_test_up = fold_test_up["Class"]
        '=======================================auc==========================================================='
        y_pred_up = prc_model.predict_proba(x_test_up)[:,1]
        y_pred_up_hm = prc_model.predict(x_test_up)
        precision_up_i,recall_up_i,threshold_up = precision_recall_curve(y_test_up,y_pred_up, pos_label=1)
        '=======================================auc==========================================================='
        auc_list = (auc(recall_i,precision_i))
        auc_list_up = (auc(recall_up_i,precision_up_i))
        del (fold_train,fold_test)
        plt.figure(1)
        plt.plot(recall_i[0:len(recall_i)-15][::-1],precision_i[0:len(precision_i)-15][::-1],color = color_origin,linestyle='-',alpha=1)
        plt.plot(recall_up_i[0:len(recall_up_i)-15][::-1],precision_up_i[0:len(precision_up_i)-15][::-1],color = color_up,linestyle='-',alpha=1)
        plt.plot(1,1,color = color_origin,label='PRC,auc:'+'%.3f'%(auc_list))
        plt.plot(1,1,color = color_up,label='PRC_up,auc:'+'%.3f'%(auc_list_up))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.legend()
        plt.legend(loc='lower left')
        plt.show()
    elif model_amount ==2:
        '==============================upsampling train/origin model_1======================='
        fold_train = pd.read_csv(file_train)
        fold_test = pd.read_csv(file_test)
        ## rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        ## upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        fold_train = shuffle(fold_train,random_state = random_state_1)
        fold_train= fold_train.reset_index(drop = True)
        ## set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
        #print('x_train:',x_train,'y_train:',y_train)
        ## set the non upsampling validation for the single fold
        fold_test = shuffle(fold_test,random_state = random_state_test)
        fold_test= fold_test.reset_index(drop = True)
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']
        '==============================upsampling train/origin model_2======================='
        fold_train_2 = pd.read_csv(file_train_2)
        #rename(消去columns項內非英文以及數字的解)
        try:
            fold_train_2 = fold_train_2.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train_2 = fold_train_2.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        #print(i,'-1')
        ## upsampling train set 
        one_message = fold_train_2[fold_train_2["Class"] == 1 ]
        zero_message =fold_train_2[fold_train_2["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train_2 = pd.concat([one_message_up,zero_message])
        fold_train_2 = shuffle(fold_train_2,random_state = random_state_2)
        fold_train_2= fold_train_2.reset_index(drop = True)
        ## set the train set for single fold
        x_train_2 = fold_train_2.drop(columns=['Class'])
        y_train_2 = fold_train_2['Class']
        #print('x_train:',x_train,'y_train:',y_train)
        ## set the non upsampling validation for the single fold
        '=============================model training=============================================================='
        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        # model_1
        prc_model = model.fit(x_train,y_train)
        y_pred = prc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
        y_pred_hm = prc_model.predict(x_test)
        precision_i , recall_i , threshold = precision_recall_curve(y_test,y_pred, pos_label=1)
        #  model_2
        prc_model_2 = model_2.fit(x_train_2,y_train_2)
        y_pred_2 = prc_model_2.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
        y_pred_hm_2 = prc_model_2.predict(x_test)
        precision_i_2 , recall_i_2 , threshold = precision_recall_curve(y_test,y_pred_2, pos_label=1)
        '==================================upsampling train/upsampling test======================================='
        fold_test_up = pd.read_csv(file_test)
        try:
            fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
        zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
        one_message_test_up = resample(one_message_test,
                replace=True,
                n_samples=len(zero_message_test),
                random_state=42)
        fold_test_up = pd.concat([one_message_test_up,zero_message_test])
        fold_test_up = shuffle(fold_test_up,random_state =random_state_test)
        fold_test_up= fold_test_up.reset_index(drop = True)
        x_test_up = fold_test_up.drop(columns = ['Class'])
        y_test_up = fold_test_up["Class"]
        '=======================================auc==========================================================='
        #model_1
        y_pred_up = prc_model.predict_proba(x_test_up)[:,1]
        y_pred_up_hm = prc_model.predict(x_test_up)
        precision_up_i,recall_up_i,threshold_up = precision_recall_curve(y_test_up,y_pred_up, pos_label=1)
        #model_2
        y_pred_up_2 = prc_model_2.predict_proba(x_test_up)[:,1]
        y_pred_up_hm_2 = prc_model_2.predict(x_test_up)
        precision_up_i_2,recall_up_i_2,threshold_up = precision_recall_curve(y_test_up,y_pred_up_2, pos_label=1)
        '=======================================auc==========================================================='
        auc_list = (auc(recall_i,precision_i))
        auc_list_up = (auc(recall_up_i,precision_up_i))
        auc_list_2 = (auc(recall_i_2,precision_i_2))
        auc_list_up_2 = (auc(recall_up_i_2,precision_up_i_2))
        del (fold_train,fold_test)
        plt.figure(1)
        plt.plot(recall_i[0:len(recall_i)-15][::-1],precision_i[0:len(precision_i)-15][::-1],color = color_origin,linestyle='-',alpha=1)
        plt.plot(recall_up_i[0:len(recall_up_i)-15][::-1],precision_up_i[0:len(precision_up_i)-15][::-1],color = color_up,linestyle='-',alpha=1)
        plt.plot(recall_i_2[0:len(recall_i_2)-15][::-1],precision_i_2[0:len(precision_i_2)-15][::-1],color = color_origin_2,linestyle='-',alpha=1)
        plt.plot(recall_up_i_2[0:len(recall_up_i_2)-15][::-1],precision_up_i_2[0:len(precision_up_i_2)-15][::-1],color = color_up_2,linestyle='-',alpha=1)
        plt.plot(1,1,color = color_origin,label='PRC,auc:'+'%.3f'%(auc_list))
        plt.plot(1,1,color = color_up,label='PRC_up,auc:'+'%.3f'%(auc_list_up))
        plt.plot(1,1,color = color_origin_2,label='PRC,auc:'+'%.3f'%(auc_list_2))
        plt.plot(1,1,color = color_up_2,label='PRC_up,auc:'+'%.3f'%(auc_list_up_2))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.legend()
        plt.legend(loc='lower left')
        plt.show()


def test_roc(
    model =None,
    model_2 = None,
    model_amount = 1,
    color_origin='orange',
    color_up = 'red',
    color_origin_2='green',
    color_up_2='green',
    file_train = None,
    file_test = None,
    file_train_2 = None,
    random_state_1 = 42,
    random_state_2 = 42,
    random_state_test = 42,
    ):
    '''
    this code not been use anymore
    '''
    fper = []
    tper = []
    fper_up = []
    tper_up = []
    auc = []
    auc_up =[]
    fper_2 = []
    tper_2 = []
    fper_up_2 = []
    tper_up_2 = []
    auc_2 = []
    auc_up_2 =[]
    if model_amount==1:
        '==============================upsampling train/origin======================='
        fold_train = pd.read_csv(file_train)
        fold_test = pd.read_csv(file_test)
        #rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        #print(i,'-1')
        # upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        fold_train = shuffle(fold_train,random_state = random_state_1)
        fold_train= fold_train.reset_index(drop = True)
        # set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
        #print('x_train:',x_train,'y_train:',y_train)
        # set the non upsampling validation for the single fold
        fold_test = shuffle(fold_test,random_state = random_state_1)
        fold_test= fold_test.reset_index(drop = True)
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']

        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        roc_model = model.fit(x_train,y_train)
        y_pred = roc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
        y_pred_hm = roc_model.predict(x_test)
        # log train/validationS loss for non upsampling
        fper_i , tper_i , threshold = roc_curve(y_test,y_pred, pos_label=1)
        '==================================upsampling train/upsampling test======================================='
        fold_test_up = pd.read_csv(file_test)
        try:
            fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
        zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
        one_message_test_up = resample(one_message_test,
                replace=True,
                n_samples=len(zero_message_test),
                random_state=42)
        fold_test_up = pd.concat([one_message_test_up,zero_message_test])
        fold_test_up = shuffle(fold_test_up,random_state = random_state_1)
        fold_test_up= fold_test_up.reset_index(drop = True)
        x_test_up = fold_test_up.drop(columns = ['Class'])
        y_test_up = fold_test_up["Class"]
        '=======================================auc==========================================================='
        y_pred_up = roc_model.predict_proba(x_test_up)[:,1]
        y_pred_up_hm = roc_model.predict(x_test_up)
        fper_up_i,tper_up_i,threshold_up = roc_curve(y_test_up,y_pred_up, pos_label=1)

        '=======================================auc==========================================================='
        auc= (roc_auc_score(y_test,y_pred))
        auc_up = (roc_auc_score(y_test_up,y_pred_up))
        plt.plot(fper_i,tper_i,color = color_origin,linestyle='-',alpha=1)
        plt.plot(fper_up_i,tper_up_i,color = color_up,linestyle='-',alpha=1)
        plt.plot(1,1,color=color_origin,label='ROC,auc:'+'%.3f'%(auc))
        plt.plot(1,1,color = color_up,label = 'ROC_up,auc:'+'%.3f'%(auc_up))
        plt.plot([0,1],[0,1],color = 'green',linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend()
        plt.legend(loc='lower right')     
        plt.show()
    elif model_amount ==2:
        '==============================upsampling train/origin model 1======================='
        fold_train = pd.read_csv(file_train)
        #rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        # upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        fold_train = shuffle(fold_train,random_state = random_state_1)
        fold_train= fold_train.reset_index(drop = True)
        # set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
        # set test
        fold_test = pd.read_csv(file_test)
        try:
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        #rename(消去columns項內非英文以及數字的解)
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = shuffle(fold_test,random_state = random_state_test)
        fold_test= fold_test.reset_index(drop = True)
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']
        '===================================upsampling train/origin model 2==========================================='
        fold_train_2 = pd.read_csv(file_train_2)
        #rename(消去columns項內非英文以及數字的解)
        try:
            fold_train_2 = fold_train_2.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train_2 = fold_train_2.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        # upsampling train set 
        one_message = fold_train_2[fold_train_2["Class"] == 1 ]
        zero_message =fold_train_2[fold_train_2["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train_2 = pd.concat([one_message_up,zero_message])
        fold_train_2 = shuffle(fold_train_2,random_state = random_state_2)
        fold_train_2 = fold_train_2.reset_index(drop = True)
        # set the train set for single fold
        x_train_2 = fold_train_2.drop(columns=['Class'])
        y_train_2 = fold_train_2['Class']
        #print('x_train:',x_train,'y_train:',y_train)
        '======================================roc======================================================='
        #model 1
        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        roc_model = model.fit(x_train,y_train)
        y_pred = roc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
        y_pred_hm = roc_model.predict(x_test)
        #log train/validationS loss for non upsampling
        fper_i , tper_i , threshold = roc_curve(y_test,y_pred, pos_label=1)
        # model 2
        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        roc_model_2 = model_2.fit(x_train_2,y_train_2)
        y_pred_2 = roc_model_2.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖
        y_pred_hm_2 = roc_model_2.predict(x_test)
        #log train/validationS loss for non upsampling
        fper_i_2 , tper_i_2 , threshold = roc_curve(y_test_2,y_pred_2, pos_label=1)

        '==================================upsampling train/upsampling test======================================='
        fold_test_up = pd.read_csv(file_test)
        fold_test_up = fold_test_up.drop(columns=['Unnamed: 0'])
        fold_test_up = fold_test_up.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
        zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
        one_message_test_up = resample(one_message_test,
                replace=True,
                n_samples=len(zero_message_test),
                random_state=42)
        fold_test_up = pd.concat([one_message_test_up,zero_message_test])
        fold_test_up = shuffle(fold_test_up,random_state = random_state_test)
        fold_test_up= fold_test_up.reset_index(drop = True)
        x_test_up = fold_test_up.drop(columns = ['Class'])
        y_test_up = fold_test_up["Class"]
        '=======================================roc up==========================================================='
        #model 1
        y_pred_up = roc_model.predict_proba(x_test_up)[:,1]
        y_pred_up_hm = roc_model.predict(x_test_up)
        fper_up_i,tper_up_i,threshold_up = roc_curve(y_test_up,y_pred_up, pos_label=1)
        #model 2
        y_pred_up_2 = roc_model_2.predict_proba(x_test_up)[:,1]
        y_pred_up_hm_2 = roc_model_2.predict(x_test_up)
        fper_up_i_2,tper_up_i_2,threshold_up = roc_curve(y_test_up,y_pred_up_2, pos_label=1)
        '=======================================auc==========================================================='
        auc= (roc_auc_score(y_test,y_pred))
        auc_up = (roc_auc_score(y_test_up,y_pred_up))
        auc_2= (roc_auc_score(y_test,y_pred_2))
        auc_up_2 = (roc_auc_score(y_test_up,y_pred_up_2))
        plt.plot(fper_i,tper_i,color = color_origin,linestyle='-',alpha=1)
        plt.plot(fper_up_i,tper_up_i,color = color_up,linestyle='-',alpha=1)
        plt.plot(fper_i_2,tper_i_2,color = color_origin_2,linestyle='-',alpha=1)
        plt.plot(fper_up_i_2,tper_up_i_2,color = color_up_2,linestyle='-',alpha=1)
        plt.plot(1,1,color=color_origin,label='ROC,auc:'+'%.3f'%(auc))
        plt.plot(1,1,color = color_up,label = 'ROC_up,auc:'+'%.3f'%(auc_up))
        plt.plot(1,1,color=color_origin_2,label='ROC,auc:'+'%.3f'%(auc_2))
        plt.plot(1,1,color = color_up_2,label = 'ROC_up,auc:'+'%.3f'%(auc_up_2))
        plt.plot([0,1],[0,1],color = 'green',linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend()
        plt.legend(loc='lower right')     
        plt.show()


def test_score(
    model = None,
    point_amount = 2,
    scoring = ['misclassification error'],
    seperate = False,
    space = 'geomspace',
    file_train = None,
    file_test = None,
    random_state = 42,
    ):
    '''
    model : input model type
    point_amount : how may point to do => always 2
    scoring = ['misclasssification error'] => which kind of scoring parameter
    space ='the way for training data set' => not use here
    '''
    specificity=[]
    score = pd.DataFrame(columns=[])
    score['test'] = []

    for type in scoring:
        print('start:',type)
        confuse =[]
        save = []
        fold_train = pd.read_csv(file_train)
        fold_test = pd.read_csv(file_test)
        # rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        '==============================================================================='     
        # upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        fold_train = shuffle(fold_train,random_state = random_state)
        fold_train= fold_train.reset_index(drop = True)
        # set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
        '=================================================================================='
        # set the non upsampling validation for the single fold
        fold_test = shuffle(fold_test,random_state = random_state)
        fold_test= fold_test.reset_index(drop = True)
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']
        if type != 'specificity':
        #log train/validationS loss for non upsampling
            confuse = plot_learning_curves(x_train,y_train,x_test,y_test,clf = model,point_amount =point_amount,scoring=type,space = space)#單個fold會回傳兩個list,前面那個是train set 得到的score,後面為validation得到
            # score_model = model.fit(x_train,y_train)
            # y_pred = score_model.predict(x_test)
            # precision = precision_score(y_test, y_pred)
            # print(precision)
        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        elif type =='specificity':
            spe_model = model.fit(x_train,y_train)
            y_pred = spe_model.predict(x_test)
            #log train/validationS loss for non upsampling
            #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            #specificity[i-1] = (tn / (tn+fp))
            specificity = recall_score(y_test, y_pred, pos_label=0)
        del (fold_train,fold_test)
        if type !='specificity':
            save.append(confuse[1][point_amount-2])
            score.loc['{}'.format(type)] = save
        elif type == 'specificity':
            save.append(specificity)
            #print(specificity)
            score.loc['{}'.format(type)] = save
        del(save,confuse)
        print('finished:',type)
    if seperate == True:
        return(save)
    else:
        return(score)


def test_score_up(
    model = None,
    point_amount = 2,
    scoring = ['misclassification error'],
    seperate = False,
    space = 'geomspace',
    file_train = None,
    file_test = None,
    random_state = 30,
    ):
    '''
    this function not being use any more
    
    '''
    specificity=[]
    score = pd.DataFrame(columns=[])
    score['test'] = []

    for type in scoring:
        print('start:',type)
        confuse =[]
        save = []
        fold_train = pd.read_csv(file_train)
        fold_test = pd.read_csv(file_test)
        #rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        '==============================================================================='     
        # upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        fold_train = shuffle(fold_train,random_state = random_state)
        fold_train= fold_train.reset_index(drop = True)
        # set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
        '========================================================================='
        # set the non upsampling validation for the single fold
        one_message_test = fold_test[fold_test["Class"] == 1 ]
        zero_message_test =fold_test[fold_test["Class"] == 0]
        one_message_up_test = resample(one_message_test,
                    replace=True,
                    n_samples=len(zero_message_test),
                    random_state=42)
        fold_test = pd.concat([one_message_up_test,zero_message_test])
        fold_test = shuffle(fold_test,random_state = random_state)
        fold_test= fold_test.reset_index(drop = True)         
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']
        '============================================================='
        if type != 'specificity':
        #log train/validationS loss for non upsampling
            confuse = plot_learning_curves(x_train,y_train,x_test,y_test,clf = model,point_amount =point_amount,scoring=type,space = space)#單個fold會回傳兩個list,前面那個是train set 得到的score,後面為validation得到
        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        elif type =='specificity':
            spe_model = model.fit(x_train,y_train)
            y_pred = spe_model.predict(x_test)
            #log train/validationS loss for non upsampling
            #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            #specificity[i-1] = (tn / (tn+fp))
            specificity = recall_score(y_test, y_pred, pos_label=0)
        del (fold_train,fold_test)
        if type !='specificity':
            save.append(confuse[1][point_amount-2])
            score.loc['{}'.format(type)] = save
        elif type == 'specificity':
            save.append(specificity)
            #print(specificity)
            score.loc['{}'.format(type)] = save
        del(save,confuse)
        print('finished:',type)
    if seperate == True:
        return(save)
    else:
        return(score)

def test_prc_models(
    model =list,
    random_state_1 = 30,
    random_state_test = 30,
    model_amount = 1,
    color_origin=list,
    color_up = list,
    file_train = list,
    file_test = list,
    input_type = list,
    ):
        '''
        model : input model list
        color_origin: input color list (for origin)
        color_up :input color list(for upsample)
        input_type :for the name of your input data
        file_train : list including path for your training data
        file_test :list including path for your testing data 
        '''
        for i in range(model_amount):
                print('now in:',i+1)
                '==============================upsampling train/origin model_1======================='
                fold_train = pd.read_csv(file_train[i])
                fold_test = pd.read_csv(file_test[i])
                #rename(消去columns項內非英文以及數字的解)
                try:
                    fold_train = fold_train.drop(columns=['Unnamed: 0'])
                    fold_test = fold_test.drop(columns=['Unnamed: 0'])
                except:
                    pass
                fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
                fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
                #print(i,'-1')
                # upsampling train set 
                one_message = fold_train[fold_train["Class"] == 1 ]
                zero_message =fold_train[fold_train["Class"] == 0]
                one_message_up = resample(one_message,
                                replace=True,
                                n_samples=len(zero_message),
                                random_state=42)
                fold_train = pd.concat([one_message_up,zero_message])
                fold_train = shuffle(fold_train,random_state = random_state_1)
                fold_train= fold_train.reset_index(drop = True)
                # set the train set for single fold
                x_train = fold_train.drop(columns=['Class'])
                y_train = fold_train['Class']
                #print('x_train:',x_train,'y_train:',y_train)
                # set the non upsampling validation for the single fold
                fold_test = shuffle(fold_test,random_state = random_state_test)
                fold_test= fold_test.reset_index(drop = True)
                x_test = fold_test.drop(columns=["Class"])
                y_test = fold_test['Class']
                #print(x_test)
                '=============================model training=============================================================='
                # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
                # model_1
                prc_model = model[i].fit(x_train,y_train)
                y_pred = prc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵 取為1的機率
                # np.savetxt("y_pred.txt", np.array(y_pred))
                # np.savetxt("y_test.txt", np.array(y_test))
                y_pred_hm = prc_model.predict(x_test)
                precision_i , recall_i , threshold = precision_recall_curve(y_test,y_pred, pos_label=1)
                '=======================================auc==========================================================='
                auc_list = (auc(recall_i,precision_i))
                del (fold_train,fold_test)
                plt.figure(1)
                plt.plot(recall_i[0:len(recall_i)-15][::-1],precision_i[0:len(precision_i)-15][::-1],color = color_origin[i],linestyle='--',alpha=1)
                plt.plot(1,1,color = color_origin[i],label='{}:'.format(input_type[i])+'%.3f'%(auc_list),linestyle='--')
                prc_test_data = {
                        'precision':list(precision_i),
                        'recall':list(recall_i),
                        'auc':auc_list,
                         }
                del(auc_list,recall_i,precision_i)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.legend()
        plt.legend(loc='lower left')
        plt.show()
        return(prc_test_data)


def test_roc_models(
    model =list,
    random_state_1 = 30,
    random_state_test = 30,
    model_amount = 1,
    color_origin=list,
    color_up = list,
    file_train = list,
    file_test = list,
    input_type = list,
    ):
    '''
    model : input model list
    color_origin: input color list (for origin)
    color_up :input color list(for upsample)
    input_type :for the name of your input data
    file_train : list including path for your training data
    file_test :list including path for your testing data 
    '''
    auc = []
    for i in range(model_amount):
        print('now:',i+1)
        '==============================upsampling train/origin======================='
        fold_train = pd.read_csv(file_train[i])
        fold_test = pd.read_csv(file_test[i])
        #rename(消去columns項內非英文以及數字的解)
        try:
            fold_train = fold_train.drop(columns=['Unnamed: 0'])
            fold_test = fold_test.drop(columns=['Unnamed: 0'])
        except:
            pass
        fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        #print(i,'-1')
        # upsampling train set 
        one_message = fold_train[fold_train["Class"] == 1 ]
        zero_message =fold_train[fold_train["Class"] == 0]
        one_message_up = resample(one_message,
                    replace=True,
                    n_samples=len(zero_message),
                    random_state=42)
        fold_train = pd.concat([one_message_up,zero_message])
        fold_train = shuffle(fold_train,random_state = random_state_1)
        fold_train= fold_train.reset_index(drop = True)
        # set the train set for single fold
        x_train = fold_train.drop(columns=['Class'])
        y_train = fold_train['Class']
        #print('x_train:',x_train,'y_train:',y_train)
        # set the non upsampling validation for the single fold
        fold_test = shuffle(fold_test,random_state = random_state_test)
        fold_test= fold_test.reset_index(drop = True)
        x_test = fold_test.drop(columns=["Class"])
        y_test = fold_test['Class']

        # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
        roc_model = model[i].fit(x_train,y_train)
        y_pred = roc_model.predict_proba(x_test)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖,取為1的機率
        y_pred_hm = roc_model.predict(x_test)
        #log train/validationS loss for non upsampling
        fper_i , tper_i , threshold = roc_curve(y_test,y_pred, pos_label=1)
        '=======================================auc==========================================================='
        auc= (roc_auc_score(y_test,y_pred))
        plt.plot(fper_i,tper_i,color = color_origin[i],linestyle='-',alpha=1)
        plt.plot(1,1,color=color_origin[i],label='{}:'.format(input_type[i])+'%.3f'%(auc))
        plt.plot([0,1],[0,1],color = 'green',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.legend(loc='lower right')     
    plt.show()    
    roc_test_data = {
                        'fper':list(fper_i),
                        'tper':list(tper_i),
                        'auc':auc,
                    }
    return(roc_test_data)


'''=============function for the SHAP========='''
def shapely(
    model =None,
    file_train = None,
    random_state_1 = 30,
    max_display = 27,
    file_test = None,
    solver = 'TreeExplainer',
    plot_type = None
    ):
    '''
    model -- the model we want to use (Adaboost is not supported by SHAP)
    file_train -- path for the training data
    max_display -- int: the feature we want to display on the graph
    file_test -- path for the test data 

    '''
    start_time = time.perf_counter_ns()
    '==============================upsampling train/origin======================='
    fold_train = pd.read_csv(file_train)
    fold_test = pd.read_csv(file_test)
    #rename(消去columns項內非英文以及數字的解)
    try:
        fold_train = fold_train.drop(columns=['Unnamed: 0'])
        fold_test = fold_test.drop(columns=['Unnamed: 0'])
    except:
        pass
    fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    #print(i,'-1')
    # upsampling train set 
    one_message = fold_train[fold_train["Class"] == 1 ]
    zero_message =fold_train[fold_train["Class"] == 0]
    one_message_up = resample(one_message,
                replace=True,
                n_samples=len(zero_message),
                random_state=42)
    fold_train = pd.concat([one_message_up,zero_message])
    fold_train = shuffle(fold_train,random_state = random_state_1)
    fold_train= fold_train.reset_index(drop = True)
    # set the train set for single fold
    x_train = fold_train.drop(columns=['Class'])
    y_train = fold_train['Class']
    #print('x_train:',x_train,'y_train:',y_train)
    # set the non upsampling validation for the single fold
    fold_test = shuffle(fold_test,random_state = random_state_1)
    fold_test= fold_test.reset_index(drop = True)
    x_test = fold_test.drop(columns=["Class"])
    y_test = fold_test['Class']

    # get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
    roc_model = model.fit(x_train,y_train)
    #joblib.dump(roc_model, 'D:/CODE/KD/codes/models_output/KeptGPT/lgbm/lgbm_model')
    end_time = time.perf_counter_ns()
    print('total time for training  on model:',(end_time-start_time)/1000000000)
    if solver =='Explainer':
    #solution for #xplainer
        background = shap.maskers.Independent(x_train)
        explainer = shap.Explainer(roc_model.predict,masker=background)
        shap_values = explainer(x_test) 
        #print(shap_values.shape)
        if str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            shap_values = explainer(x_test,check_additivity=False) #(for lgbm)
            # 問題點:要計算training的shap values時候，要使用oversample完後的training還是原始的資料(?)
        shap.plots.beeswarm(shap_values, max_display=max_display)   

    elif solver =='Kernel':
    # solution for Kernel
        explainer = shap.KernelExplainer(roc_model.predict_proba,data=x_train)
        shap_values = explainer.shap_values(x_test)
        shap.plots.beeswarm(shap_values, max_display=max_display)

    #solution for exception said that have more than one dimension( this happend because TreeExplainer return 3 dimension)
    elif solver =='TreeExplainer':
        explainer = shap.TreeExplainer(roc_model)  
        if str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            shap_values = explainer(x_test,check_additivity=False) #(for lgbm)
        #shap_plots = shap.plots
        if str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
            # plots = getattr(shap_plots,plot_type)
            # plots(shap_values, max_display=max_display)
            shap.plots.bar(shap_values, max_display=max_display)
        else:
            shap_values2 = copy.deepcopy(shap_values)
            shap_values2.values = shap_values2.values[:,:,1]
            shap_values2.base_values = shap_values2.base_values[:,1]
            shap_plots = shap.plots
            #plots = getattr(shap_plots,plot_type)
            #plots(shap_values, max_display=max_display)
            shap.plots.bar(shap_values2, max_display=max_display)

'''=============functions for testing datas (given models)========='''
def test_score_given_model(
    model_path = None,
    file_test = None,
    random_state_test = 30,
):
    score = pd.DataFrame(columns=[])
    score['test'] = []
    model = joblib.load(model_path)
    fold_test = pd.read_csv(file_test)
    try:
        fold_test = fold_test.drop(columns=['Unnamed: 0'])
    except:
        pass
    fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    fold_test = shuffle(fold_test,random_state = random_state_test)
    fold_test= fold_test.reset_index(drop = True)
    x_test = fold_test.drop(columns=["Class"])
    y_test = fold_test['Class']
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:,1]
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred)
    score.loc['f1'] = f1
    score.loc['precision'] = precision
    score.loc['recall'] = recall
    score.loc['specificity'] = specificity
    return score


def test_prc_given_model(
    model_path = None,
    file_test = None,
    random_state_test = 30,
):
    model = joblib.load(model_path)
    fold_test = pd.read_csv(file_test)
    try:
        fold_test = fold_test.drop(columns=['Unnamed: 0'])
    except:
        pass
    fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    fold_test = shuffle(fold_test,random_state = random_state_test)
    fold_test= fold_test.reset_index(drop = True)
    x_test = fold_test.drop(columns=["Class"])
    y_test = fold_test['Class']
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:,1]
    precision_i , recall_i , threshold = precision_recall_curve(y_test,y_pred_prob, pos_label=1)
    auc_list = (auc(recall_i,precision_i))
    plt.figure(1)
    plt.plot(recall_i[0:len(recall_i)][::-1],precision_i[0:len(precision_i)][::-1],color = 'orange',linestyle='-',alpha=1)
    plt.plot(1,1,color = 'orange',label='%.3f'%(auc_list),linestyle='-')
    prc_test_data = {
            'precision':list(precision_i),
            'recall':list(recall_i),
            'auc':auc_list,
                }
    del(auc_list,recall_i,precision_i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.legend(loc='lower left')
    plt.show()
    return(prc_test_data)


def test_roc_given_model(
    model_path = None,
    file_test = None,
    random_state_test = 30,
):
    model = joblib.load(model_path)
    fold_test = pd.read_csv(file_test)
    try:
        fold_test = fold_test.drop(columns=['Unnamed: 0'])
    except:
        pass
    fold_test = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    fold_test = shuffle(fold_test,random_state = random_state_test)
    fold_test= fold_test.reset_index(drop = True)
    x_test = fold_test.drop(columns=["Class"])
    y_test = fold_test['Class']
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:,1]
    print(y_pred_prob)
    # with open(f'{model_save_path}/pred_prob.json','w') as json_file:
    #     json.dump(y_pred_prob.tolist(), json_file)
    fper_i , tper_i , threshold = roc_curve(y_test,y_pred_prob, pos_label=1)
    '=======================================auc==========================================================='
    auc= (roc_auc_score(y_test,y_pred_prob))
    print(auc)
    plt.plot(fper_i,tper_i,color = 'orange',linestyle='-',alpha=1)
    plt.plot(1,1,color='orange',label='%.3f'%(auc))
    plt.plot([0,1],[0,1],color = 'green',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.legend(loc='lower right')     
    plt.show()    
    roc_test_data = {
                        'fper':list(fper_i),
                        'tper':list(tper_i),
                        'auc':auc,
                    }
    return(roc_test_data)

if __name__ =='__main__':
    '''models'''

    warnings.filterwarnings("ignore")
    model =  lgb.LGBMClassifier(boosting = 'gbdt',learning_rate = 0.2,random_state = 42,min_data_in_leaf = 10,max_depth = 5,num_leaves = 5)
    #model = LDA(solver = 'svd')
    #model = LinearRegression()
    #model = xgb.XGBClassifier(random_state = 42,max_depth = 5,learning_rate =2,reg_lambda = 60000)#,gamma = 100) 
    #model = AdaBoostClassifier(n_estimators = 2500,random_state = 42,learning_rate = 0.05)
    #model = RandomForestClassifier(criterion = 'gini',max_depth =5,n_estimators = 100, min_samples_leaf= 10,max_features = 'log2',max_samples = 0.5,random_state=42)

    #model = LogisticRegression(random_state=42,penalty='none',C=4000,max_iter=7000,solver='lbfgs',warm_start=True)
    #model  = MLPClassifier(random_state=42,learning_rate_init=0.001,max_iter=130,alpha = 0.7,solver = 'adam',activation='relu',)
    #model = svm.SVC(kernel = 'rbf', C = 0.95, gamma = 1e-2,max_iter = 3000,random_state = 42,tol =1e-3,probability=True,cache_size=2000)# (稍微還是有點overfitting) 


    '''for SHAP'''
    #path = 'D:/CODE/KD/code_after_web/data/KeptGPT_under7/k-fold/30/5-fold_train_z.csv'
    #path_test = 'D:/CODE/KD/code_after_web/data/KeptGPT_under7/k-fold/30/total-test_z-fold5.csv'
    #shapely(model=model,file_train=path,random_state_1=30,max_display=10,file_test=path_test,plot_type='bar')

    '''for validation'''
    path = '/home/cosbi/KDGPT/data_processing/haung_kfold_output' 
    model_name = 'lgbm'
    model_save_path = f'/home/cosbi/KDGPT/model_training/model_save/{model_name}'
    if os.path.exists(model_save_path) == False:
        os.makedirs(model_save_path)

    ### train_loss,test_loss,train_loss_up,test_loss_up,rng = kfold_self(model = model,point_amount=18,random_state=30,path = path,space='geomspace',loss_log=True)#,scoring='f1') # for 顯示每個fold 的訓練結果
    
    # f1 = kfold_self(model = model,point_amount=18,random_state=30,path = path,space='geomspace',loss_log=False,scoring='f1') #for only顯示average,max,min
    # f1_json = json.dumps(f1)
    # with open(f'{model_save_path}/f1.json','w') as json_file:
    #     json_file.write(f1_json)

    # loss = kfold_self(model = model,point_amount=18,random_state=30,path = path,space='geomspace',loss_log=False) #for only顯示average,max,min
    # loss_json = json.dumps(loss)
    # with open(f'{model_save_path}/loss.json','w') as json_file:
    #     json_file.write(loss_json)
    # print(loss_json)

    # score = kfold_self_score(path =path,model = model,scoring=['f1','precision','recall','specificity'],seperate = False,random_state=30,model_name = model_name)
    # score.to_csv(f'{model_save_path}/{model_name}_spe_validation.csv')
    # score_up = kfold_self_score_up(path = path,model = model,scoring=['f1','precision','recall','specificity'],seperate = False,random_state=30,model_name = model_name)
    # score_up.to_csv(f'{model_save_path}/{model_name}_score_up_spe_validation.csv')
    # print(score)

    # ''' PRC / ROC裡面有drop model(要改路徑)'''
    # prc_curve_data = prc_plot(path=path,random_state_1=30,model = model,model_amount=1,color_origin='orange',color_up='red',model_save_path = model_save_path)
    # prc_json = json.dumps(prc_curve_data)
    # with open(f'{model_save_path}/prc_validation.json','w') as json_file:
    #   json_file.write(prc_json)

    # roc_curve_data = roc_plot(path=path,random_state_1=30,model = model,model_amount=1,color_origin='orange',color_up='red',color_origin_2='blue',color_up_2='skyblue')
    # roc_json = json.dumps(roc_curve_data)
    # with open(f'{model_save_path}/roc_validation.json','w') as json_file:
    #   json_file.write(roc_json)


    '''for test'''
    '''----given model -----'''
    model_name = 'lgbm'
    best_model_num = 3
    model_save_path = f'/home/cosbi/KDGPT/model_training/model_save/{model_name}'
    model_path = f'{model_save_path}/{best_model_num}-model'
    file_test = f'/home/cosbi/KDGPT/data_processing/haung_kfold_output/total-test_z-fold{best_model_num}.csv' 

    score = test_score_given_model(model_path=model_path,file_test=file_test,random_state_test = 30)
    score.to_csv(f'{model_save_path}/score_test.csv')
    print(score)
    
    prc_test_data = test_prc_given_model(model_path=model_path,file_test=file_test,random_state_test = 30)
    prc_test_json = json.dumps(prc_test_data)
    with open(f'{model_save_path}/prc_test.json','w') as json_file:
       json_file.write(prc_test_json)

    roc_test_data = test_roc_given_model(model_path=model_path,file_test=file_test,random_state_test = 30)
    roc_test_data = json.dumps(roc_test_data)
    with open(f'{model_save_path}/roc_test.json','w') as json_file:
       json_file.write(roc_test_data)

    '''-------not given model-------'''
    # file_test = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/lam_both/total-test_1_z.csv'
    # file_train_2 ='D:/CODE/KD/datas/KeptGPT/k-fold/30/lam_both/1-fold_train_z.csv'
    # score = test_score(model = model,scoring=['f1','precision','recall','specificity'],random_state=30,seperate = False,space='geomspace',file_test=file_test,file_train=file_train_2)
    # score.to_csv('models_output/KeptGPT/MLP/sgd_score_test.csv')

    
    #file_train = 'D:/CODE/KD/code_after_web/data/KeptGPT/without gender/k-fold/30/3-fold_train_z.csv'
    #file_test = 'D:/CODE/KD/code_after_web/data/KeptGPT/without gender/k-fold/30/total-test_z-fold3.csv'
    #prc_test_data = test_roc_models(model=[model],color_origin=['orange'],color_up=['red'],input_type=['lgbm'],file_train=[file_train],file_test=[file_test],random_state_1=30,random_state_test=30)
    # prc_test_json = json.dumps(prc_test_data)
    # with open('D:/CODE/KD/codes/models_output/Top5/lgbm/Top5_prc_test_30.json','w') as json_file:
    #    json_file.write(prc_test_json)
    
    # scoring=['f1','precision','recall','specificity']
    # score = test_score(model = model,scoring=['f1','precision','recall','specificity'],random_state=30,seperate = False,file_test=file_test,file_train=file_train)
    # score.to_csv('D:/CODE/KD/codes/models_output/Top5/lgbm/lgbm_score_spe_test.csv')
    #print(score)


    #linear_test(model=[model],color_origin=['orange'],input_type=['Linear Regression'],file_train=[file_train],file_test=[file_test],change_column='EOSpzpscore')
    


    