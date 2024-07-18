import pandas as pd
from sklearn.model_selection import train_test_split,KFold
import numpy as np
import re
import os
from tqdm import trange
import argparse

def split_kfold(raw_df ,cv = 5,random_state = 42,folderpath ='k_fold_train_validation_42'):
    '''
    raw_df: DataFrame -- the dataframe that contain the data before seperate in to train and test set
    cv : int -- to choose how many fold we want to split
    random_state: int -- to set the random seed for split(including train/test and train/validation this two in here share the same random seed)
    '''
    if os.path.isdir(folderpath):
        pass
    else:
        os.makedirs(folderpath)
    x = raw_df.drop(columns = ['Class'])
    y = raw_df[['Class']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    test_df = pd.DataFrame()
    test_df = X_test.copy()
    test_df["Class"] = y_test
    train_df = pd.DataFrame()
    train_df = X_train.copy() #這個很重要 要用copy複製內容而不是直接導向同一個記憶體位置
    train_df["Class"] = y_train
    test_df.to_csv(folderpath+'/total-test.csv',index = False)
    train_df.to_csv(folderpath+'/total-train-val.csv',index =False)
    #這邊切分的是5-fold 要用的訓練以及validation集
    #np.set_printoptions(threshold=sys.maxsize)
    train_train = []
    test_test = []
    kf = KFold(n_splits=cv,shuffle=True ,random_state=random_state)
    train_df_np = train_df.to_numpy()
    folded = list(kf.split(X = train_df_np)) #把訓練資料分成K筆，後續要再度將K-1比訓練資料做oversampling，另一組做oversamplin & origin當作validation，重複K次

    for i in range (len(folded)):
        train_train.append(folded[i][0])
        test_test.append(folded[i][1])
    for fold in trange(len(folded)):
        #print('fold:',fold)
        train_fold = pd.DataFrame(columns=list(raw_df.columns))
        test_fold = pd.DataFrame(columns=list(raw_df.columns))

        for i in train_train[fold]:
            #print('train:',i)
            train_fold.loc[len(train_fold)] = train_df_np[i].tolist()
        for i in test_test[fold]:
            #print('validation:',i)
            test_fold.loc[len(test_fold)] = train_df_np[i].tolist()

        test_fold.to_csv(folderpath +'/{}-fold_test.csv'.format(fold+1),index=False)
        train_fold.to_csv(folderpath+'/{}-fold_train.csv'.format(fold+1),index=False)
        
        del(train_fold,test_fold)
    return(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    '''=================================KD / FC====================================='''
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1",
                    help="請輸入第一個輸入檔案(.csv)")
    parser.add_argument("--output1",
                    help="請輸入第一個輸出資料夾路徑")
    args = parser.parse_args()
    raw_df = pd.read_csv(args.input1)
    #''' for all features/ drop ALT/drop AST
    raw_df = raw_df.drop(columns = ['ID'])
    #'''


    raw_df = raw_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', 'p', x))
    split_kfold(raw_df = raw_df,random_state= 30,folderpath = args.output1)


