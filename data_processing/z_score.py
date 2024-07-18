from operator import index
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import re
def z_score(path,output = True):
    '''
    path: giving the path of the .csv file,
    output: default = True
        if output set as True, will out put a .csv file name as {import file name}_z.csv 
        if output set as False, will not out put any data
    The function return fit transform
    '''
    data = pd.read_csv(path)

    data_par = data.drop(columns=['Class'])
    try:
        data_par = data_par.drop(columns=['Unnamed: 0'])
    except:
        pass
    data_class = data["Class"]
    for parameter in data_par:
        data_par = data_par.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
        data_par['{}'.format(parameter)] = data_par['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
        data_par = data_par.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float


    scale = StandardScaler() #z-scaler物件
    train_set_scaled = pd.DataFrame(scale.fit_transform(data_par),
                                    columns=data_par.keys()) #用standardscaler +fit_transform將column轉成z_score,fit是代表找到特性(mean,std)，transform轉成我們要的scale
    train_set_scaled["Class"] = data_class
    train_set = scale.fit(data_par)
    train_mean_var = pd.DataFrame(columns = list(data_par.columns))
    train_mean_var.loc['mean'] = train_set.mean_
    train_mean_var.loc['std'] = (train_set.var_)**0.5
    #display(train_set_scaled)
    output_path = path.replace('.csv','')
    if output == True:
        train_set_scaled.to_csv('{}_z.csv'.format(output_path),index=False)
        train_mean_var.to_csv('{}_mean_std.csv'.format(output_path),index=False)
    else:
        pass
    return(train_set_scaled,train_mean_var)


def z_score_test(path_mean_std,path_test,output = True):
    '''
    # 會保留class欄位
    path_mean_std:given the dir for the place that save mean_std value
    path:dir that we want to out put the solution
    '''
    mean_std = pd.read_csv(path_mean_std)
    test = pd.read_csv(path_test)
    try:
        mean_std = mean_std.drop(columns=['Unnamed: 0'])
    except:
        pass

    for parameter in mean_std:
        mean_std = mean_std.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
        mean_std['{}'.format(parameter)] = mean_std['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
        mean_std = mean_std.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float



    test_class = test['Class']

    test = test.drop(columns=['Class'])
    try:
        test =test.drop(columns=['Unnamed: 0'])
    except:
        pass
    for parameter in test:
        test = test.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
        test['{}'.format(parameter)] = test['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
        test = test.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float

    z_score = pd.DataFrame()
    for i in test:
        z_score[i] = ((test[i]-mean_std[i][0])/mean_std[i][1])
    #print(display(z_score))
    z_score['Class'] = test_class
    output_path = path_test.replace('.csv','')
    if output ==True:
        z_score.to_csv('{}_z.csv'.format(output_path),index=False)
    else:
        pass
    return(z_score)

def z_score_test_real_use(path_mean_std,path_test,output = True):
    '''
    # 在實際上線時使用(無class欄位)
    path_mean_std:given the dir for the place that save mean_std value
    path:dir that we want to out put the solution
    '''
    mean_std = pd.read_csv(path_mean_std)
    test = pd.read_csv(path_test)
    try:
        mean_std = mean_std.drop(columns=['Unnamed: 0'])
    except:
        pass

    for parameter in mean_std:
        mean_std = mean_std.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
        mean_std['{}'.format(parameter)] = mean_std['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
        mean_std = mean_std.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float
    try:
        test = test.drop(columns=['Unnamed: 0'])
    except:
        pass

    for parameter in test:
        test = test.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
        test['{}'.format(parameter)] = test['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
        test = test.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float

    z_score = pd.DataFrame()
    for i in test:
        z_score[i] = ((test[i]-mean_std[i][0])/mean_std[i][1])
    #print(display(z_score))
    output_path = path_test.replace('.csv','')
    if output ==True:
        z_score.to_csv('{}_z.csv'.format(output_path),index=False)
    else:
        pass
    return(z_score)

if __name__ == '__main__':

    # for i in range(1,6):
    #     path = './lam_kfold_output/{}-fold_train.csv'.format(i)
    #     origin_data = pd.read_csv(path)
    #     train_set,train_mean_var= z_score(path = path,output = False)
    #     # train_set[['EOSpzpscore']] = origin_data[['EOSpzpscore']]
    #     #train_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = origin_data[['HBpzpscore','EOSpzpscore','PLTpzpscore']]
    #     output_path = path.replace('.csv','')
    #     train_set.to_csv('{}_z.csv'.format(output_path),index=False)
    #     train_mean_var.to_csv('{}_mean_std.csv'.format(output_path),index=False)
    # ##example for useing the z_score function
    
    # for i in range(1,6):
    #     path_mean_std = './lam_kfold_output/{}-fold_train_mean_std.csv'.format(i)
    #     path_test = './lam_kfold_output/{}-fold_test.csv'.format(i)
    #     origin_data = pd.read_csv(path_test)
    #     test_set = z_score_test(path_mean_std=path_mean_std,path_test = path_test,output=False)
    #     # test_set[['EOSpzpscore']] = origin_data[['EOSpzpscore']]
    #     #test_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = origin_data[['HBpzpscore','EOSpzpscore','PLTpzpscore']]
    #     output_path = path_test.replace('.csv','')
    #     test_set.to_csv('{}_z.csv'.format(output_path),index=False)
    
     
##################################################################################################################################################################################
    # example for using the z_score_test function
    
    fold = 3
    path_mean_std = f'./haung_kfold_output/{fold}-fold_train_mean_std.csv'
    path_test = './haung_kfold_output/total-test.csv'
    origin_data = pd.read_csv(path_test)
    test_set = z_score_test(path_mean_std=path_mean_std,path_test = path_test,output=False)
    # test_set[['EOSpzpscore']] = origin_data[['EOSpzpscore']]
    #test_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = origin_data[['HBpzpscore','EOSpzpscore','PLTpzpscore']]
    output_path = path_test.replace('.csv','')
    test_set.to_csv('{}_z-fold{}.csv'.format(output_path,fold),index = False)
    
    
##################################################################################################################################################################################
#no used   
    ''' for no z score
    for i in range(1,6):
        path = 'D:/CODE/KD/code_after_web/data/lam/k-fold/1200/{}-fold_train.csv'.format(i)
        origin_data = pd.read_csv(path)
        train_set,train_mean_var= z_score(path = path,output = False)
        output_path = path.replace('.csv','')
        train_set.to_csv('{}_z.csv'.format(output_path),index=False)
        train_mean_var.to_csv('{}_mean_std.csv'.format(output_path),index=False)
    for i in range(1,6):
        path_mean_std = 'D:/CODE/KD/code_after_web/data/lam/k-fold/1200/{}-fold_train_mean_std.csv'.format(i)
        path_test = 'D:/CODE/KD/code_after_web/data/lam/k-fold/1200/{}-fold_test.csv'.format(i)
        origin_data = pd.read_csv(path_test)
        test_set = z_score_test(path_mean_std=path_mean_std,path_test = path_test,output=False)
        output_path = path_test.replace('.csv','')
        test_set.to_csv('{}_z.csv'.format(output_path),index=False)
    '''

    '''
    for i in range(1,6):
        path = 'D:/CODE/KD/code_after_web/data/Top5/k-fold/30/{}-fold_train.csv'.format(i)
        origin_data = pd.read_csv(path)
        train_set,train_mean_var= z_score(path = path,output = False)
        #train_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = origin_data[['HBpzpscore','EOSpzpscore','PLTpzpscore']]
        train_set[['EOSpzpscore']] = origin_data[['EOSpzpscore']]
        output_path = path.replace('.csv','')
        train_set.to_csv('{}_z.csv'.format(output_path),index=False)
        train_mean_var.to_csv('{}_mean_std.csv'.format(output_path),index=False)
    # example for useing the z_score function
    
    for i in range(1,6):
        path_mean_std = 'D:/CODE/KD/code_after_web/data/Top5/k-fold/30/{}-fold_train_mean_std.csv'.format(i)
        path_test = 'D:/CODE/KD/code_after_web/data/Top5/k-fold/30/{}-fold_test.csv'.format(i)
        origin_data = pd.read_csv(path_test)
        test_set = z_score_test(path_mean_std=path_mean_std,path_test = path_test,output=False)
        #test_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = origin_data[['HBpzpscore','EOSpzpscore','PLTpzpscore']]
        test_set[['EOSpzpscore']] = origin_data[['EOSpzpscore']]
        output_path = path_test.replace('.csv','')
        test_set.to_csv('{}_z.csv'.format(output_path),index=False)
    '''


    '''for single file
    path = 'D:/CODE/KD/datas/KeptGPT/kd_keptGPT.csv'
    origin_data = pd.read_csv(path,index_col=[0])
    origin_data = origin_data.drop(columns = ['age (day)','age (year)','ID','性別'])
    origin_data = origin_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', 'p', x))
    origin_data.to_csv('D:/CODE/KD/datas/KeptGPT/kd_KeptGPT_for_zscore.csv')
    path2 = 'D:/CODE/KD/datas/KeptGPT/kd_KeptGPT_for_zscore.csv'
    train_set,train_mean_var= z_score(path = path2,output = False)
    train_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = origin_data[['HBpzpscore','EOSpzpscore','PLTpzpscore']]
    #train_set[['EOSpzpscore','PLTpzpscore']] = origin_data[['EOSpzpscore','PLTpzpscore']]
    output_path = path.replace('.csv','')
    #train_set.to_csv('{}_z.csv'.format(output_path))
    train_mean_var.to_csv('{}_mean_std.csv'.format(output_path))
    '''
    '''example for using the z_score_test function
    path_mean_std = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/5-fold_train_mean_std.csv'
    path_test = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/total-test.csv'
    origin_data = pd.read_csv(path_test)
    test_set = z_score_test(path_mean_std=path_mean_std,path_test = path_test,output=False)
    test_set[['HBpzpscore','EOSpzpscore','PLTpzpscore']] = origin_data[['HBpzpscore','EOSpzpscore','PLTpzpscore']]
    output_path = path_test.replace('.csv','')
    test_set.to_csv('{}_5_z.csv'.format(output_path))
    '''
    


