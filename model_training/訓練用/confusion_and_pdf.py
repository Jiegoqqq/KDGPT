import warnings
import json
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,recall_score,precision_recall_curve,auc,precision_score,f1_score
from sklearn.utils import resample,shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from learn_curving_fold import plot_learning_curves
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import copy
#from svm-gpu import SVM
from IPython.display import display
import os
import joblib
from tqdm import trange



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
    return -- the accuracy or other parameter in confusion matrix for the non upsampling validation set
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
                y_pred_prob = spe_model.predict_proba(x_test)
                y_pred_train = spe_model.predict(x_train)
                y_pred_train_prob = spe_model.predict_proba(x_train)
                if y_pred.ndim != 1:
                    y_pred = y_pred.reshape(y_pred.shape[0],)
                ## log train/validationS loss for non upsampling
                ## tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                # specificity[i-1] = (tn / (tn+fp))
                specificity[i-1] = recall_score(y_test, y_pred, pos_label=0)
                total_pred = [
                    {'train':y_train.tolist()},
                    {'validation':y_test.tolist()},
                    {'y_pred_val':y_pred.tolist()},
                    {'y_pred_prob_val':y_pred_prob.tolist()},
                    {'y_pred_train':y_pred_train.tolist()},
                    {'y_pred_prob_train':y_pred_train_prob.tolist()}
                    ]
                total_pred_json = json.dumps(total_pred)
                with open(f'{model_save_path}/train_vlidation_pred_{i}.json','w') as json_file:
                    json_file.write(total_pred_json)
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
    
def test_score_given_model(
    model_path = None,
    file_test = None,
    random_state_test = 30,
    model_save_path=None
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
    y_pred_prob = model.predict_proba(x_test)
    total_pred = [
    {'test':y_test.tolist()},
    {'y_pred_test':y_pred.tolist()},
    {'y_pred_prob_test':y_pred_prob.tolist()},
    ]
    total_pred_json = json.dumps(total_pred)
    with open(f'{model_save_path}/train_origin_pred_3.json','w') as json_file:
        json_file.write(total_pred_json)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred)
    score.loc['f1'] = f1
    score.loc['precision'] = precision
    score.loc['recall'] = recall
    score.loc['specificity'] = specificity
    return score

def test_score_up_given_model(
    model_path = None,
    file_test = None,
    random_state_test = 30,
    model_save_path=None,
    file_name=None,
):
    score = pd.DataFrame(columns=[])
    score['test'] = []
    model = joblib.load(model_path)
    fold_test = pd.read_csv(file_test)
    try:
        fold_test = fold_test.drop(columns=['Unnamed: 0'])
    except:
        pass
    fold_test_up = fold_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    one_message_test = fold_test_up[fold_test_up["Class"] == 1 ]
    zero_message_test =fold_test_up[fold_test_up["Class"] == 0]
    one_message_test_up = resample(one_message_test,
                replace=True,
                n_samples=len(zero_message_test),
                random_state=42)
    fold_test_up = pd.concat([one_message_test_up,zero_message_test])
    fold_test_up = shuffle(fold_test_up,random_state = random_state_test)
    fold_test_up = fold_test_up.reset_index(drop = True)
    x_test_up = fold_test_up.drop(columns=["Class"])
    y_test_up = fold_test_up['Class']
    y_pred = model.predict(x_test_up)
    y_pred_prob = model.predict_proba(x_test_up)
    total_pred = [
    {'test':y_test_up.tolist()},
    {'y_pred_test':y_pred.tolist()},
    {'y_pred_prob_test':y_pred_prob.tolist()},
    ]
    total_pred_json = json.dumps(total_pred)
    with open(f'{model_save_path}/{file_name}.json','w') as json_file:
        json_file.write(total_pred_json)
    f1 = f1_score(y_test_up, y_pred)
    specificity = recall_score(y_test_up, y_pred, pos_label=0)
    recall = recall_score(y_test_up, y_pred, pos_label=1)
    precision = precision_score(y_test_up, y_pred)
    score.loc['f1'] = f1
    score.loc['precision'] = precision
    score.loc['recall'] = recall
    score.loc['specificity'] = specificity
    return score

def confusion_matrix(true,pred):
    TN = 0
    TP = 0
    FP = 0
    FN = 0 
    for i in trange(len(true)):
        if true[i] == pred[i] and pred[i] == 1:
            TP += 1 
        elif true[i] != pred[i] and pred[i] == 1:
            FP += 1
        elif true[i] == pred[i] and pred[i] == 0:
            TN += 1 
        elif true[i] != pred[i] and pred[i] == 0:
            FN += 1
    return(TP,TN,FP,FN)

def roc_plot_training(
    model = None,
    fold = 5,
    path = 'k_fold_train_validation_42',
    random_state_1=30,
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
    roc_dict = {}
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
        y_pred = roc_model.predict_proba(x_train)[:,1]# 注意這種寫法的意涵，還有ROC 要用機率出圖,取為1的機率
        ## log train/validationS loss for non upsampling
        fper_i , tper_i , threshold = roc_curve(y_train,y_pred, pos_label=1)
        roc_dict[f'{i}_fold_fper_tper'] = [fper_i.tolist(),tper_i.tolist(),roc_auc_score(y_train,y_pred)]
        '=======================================auc==========================================================='
    del (fold_train,fold_test)
    return(roc_dict)  

if __name__ =='__main__':
    warnings.filterwarnings("ignore")

    '''=================train up============='''
    # model =  lgb.LGBMClassifier(boosting = 'gbdt',learning_rate = 0.2,random_state = 42,min_data_in_leaf = 10,max_depth = 5,num_leaves = 5)
    # path = 'D:/CODE/KD/code_after_web/data/KeptGPT_under7(ByKeptGPT)/k-fold_monocyte_count/30' 
    # model_name = 'lgbm'
    # model_save_path = f'D:/CODE/KD/code_after_web/output/Under 7/KeptGPT_under7(ByKeptGPT,with monocyte count)/{model_name}'
    # score = kfold_self_score(path =path,model = model,scoring=['specificity'],seperate = False,random_state=30,model_name = model_name,file_name = 'train_up_predict')
    # print(score)
    '''================train origin========='''
    # model_name = 'lgbm'
    # best_model_num = 3
    # model_save_path = f'D:/CODE/KD/code_after_web/output/Under 7/KeptGPT_under7(ByKeptGPT,with monocyte count)/{model_name}'
    # model_path = f'{model_save_path}/{best_model_num}-model'
    # file_test = f'D:/CODE/KD/code_after_web/data/KeptGPT_under7(ByKeptGPT)/k-fold_monocyte_count/30/{best_model_num}-fold_train_z.csv' 
    # score = test_score_given_model(model_path=model_path,file_test=file_test,random_state_test = 30,model_save_path = model_save_path,file_name = 'train_predict')
    # print(score)
    
    '''==============test / validation_up================='''
    # model_name = 'lgbm'
    # best_model_num = 3
    # model_save_path = f'D:/CODE/KD/code_after_web/output/without gender/{model_name}'
    # model_path = f'{model_save_path}/{best_model_num}-model'
    # file_test = f'D:/CODE/KD/code_after_web/data/KeptGPT/without gender/k-fold/30/total-test_z-fold{best_model_num}.csv' 
    # score = test_score_given_model(model_path=model_path,file_test=file_test,random_state_test = 30,model_save_path = model_save_path,file_name='test_predict')
    # print(score)
    


    '''====================confusion==============='''
    ## Training
    # with open('D:/CODE/KD/code_after_web/output/Under 7/KeptGPT_under7(ByKeptGPT,with monocyte count)/lgbm/train_predict.json') as json_file: 
    #     data_train = json.load(json_file)
    # data_dict_train = {}
    # for i in data_train:
    #     print(list(i.keys())[0])
    #     data_dict_train[list(i.keys())[0]] = list(i.values())[0]
    # kd_train = []
    # fc_train = []
    # for i in range(len(data_dict_train['y_pred_prob_train'])):
    #     if data_dict_train['train'][i] == 1:
    #         kd_train.append(data_dict_train['y_pred_prob_train'][i][1])
    #     elif data_dict_train['train'][i] == 0:
    #         fc_train.append(data_dict_train['y_pred_prob_train'][i][1])
    # print(len(kd_train),len(fc_train))
    # TP,TN,FP,FN = confusion_matrix(data_dict_train['train'],data_dict_train['y_pred_train'])
    # print('Training--','\tTP:',TP,'\tTN:',TN,'\tFP:',FP,'\tFN:',FN)
    # sns.kdeplot(kd_train,  color = 'red',label = 'KD Training')
    # sns.kdeplot(fc_train,  color = 'blue',label = 'FC Training')
    # plt.xlim([-0.5,1.5])
    # plt.legend(loc='upper center')
    # plt.show()

    # ### Test
    with open('D:/CODE/KD/code_after_web/output/Under 7/KeptGPT_under7(ByKeptGPT,with monocyte count)/lgbm/test_predict.json') as json_file: 
        data_test = json.load(json_file)
    data_dict_test = {}
    for i in data_test:
        data_dict_test[list(i.keys())[0]] = list(i.values())[0]
    kd_test = []
    fc_test = []
    for i in range(len(data_dict_test['y_pred_prob_test'])):
        if data_dict_test['test'][i] == 1:
            kd_test.append(data_dict_test['y_pred_prob_test'][i][1])
        elif data_dict_test['test'][i] == 0:
            fc_test.append(data_dict_test['y_pred_prob_test'][i][1])
    print(len(kd_test),len(fc_test))
    TP_t,TN_t,FP_t,FN_t = confusion_matrix(data_dict_test['test'],data_dict_test['y_pred_test'])
    print('Test--','\tTP:',TP_t,'\tTN:',TN_t,'\tFP:',FP_t,'\tFN:',FN_t)
    sns.kdeplot(kd_test,  color = 'red',label = 'KD Test')
    sns.kdeplot(fc_test,  color = 'blue',label = 'FC Test')
    plt.xlim([-0.5,1.5])
    plt.legend(loc='upper center')
    plt.show()


