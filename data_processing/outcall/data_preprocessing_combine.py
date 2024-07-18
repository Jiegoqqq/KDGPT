
from cmath import nan
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from IPython.display import display 
import os 
import argparse

def drop_all(fc,kd):
    for parameter in fc:
        fc['{}'.format(parameter)].apply(lambda x:np.NaN if str(x).isspace() else x)
        kd['{}'.format(parameter)].apply(lambda x:np.NaN if str(x).isspace() else x)
    fc = fc.dropna()
    kd = kd.dropna()
    i = 0
    for parameter in fc:
        if parameter == 'ID':
            pass
        else:
            try:
                fc = fc.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
                fc['{}'.format(parameter)] = fc['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
                fc = fc.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float

                kd = kd.astype({'{}'.format(parameter):'str'})
                kd['{}'.format(parameter)] = kd['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)
                kd = kd.astype({'{}'.format(parameter):'float'})            
                i+=1
            except Exception as e:
                print('{}'.format(parameter),e)
                pass
    fc.insert(i+1,'Class', 0)
    kd.insert(i+1,'Class', 1)
    #輸出兩個dataframe觀察是否有錯誤
    fc.to_csv('../datas/fc.csv')
    kd.to_csv('../datas/kd.csv')
    #將兩dataFrame合而為一
    combine = pd.concat([fc,kd],axis = 0)#axis = 0時為縱向合併
    combine.set_index('ID',inplace = True)
    combine.to_csv('../datas/combine_fc_kd.csv')
    print(fc.shape)
    print(kd.shape)
    print(combine.shape)

    
def fill0_select(select = list,fc=nan,kd=nan):
    for feature in select:
        kd[feature] =kd[feature].fillna(0)
        fc[feature] = fc[feature].fillna(0)
    return(fc,kd)
def count_na(select = list,fc=None,kd = None):
    fc_nan = {}
    kd_nan = {}
    for feature in select:
        fc_nan[feature] = fc[feature].isna().sum()
        kd_nan[feature] = kd[feature].isna().sum()
    return (fc_nan,kd_nan)



if __name__=='__main__':
    '''
    處理順序:
    1.先填補整個檔案的空值(如果要填補的 依照規則有值)
    2.抽取需要的欄位
    3.若需要的欄位內有病患經過步驟一之後依然有空值，則丟棄病患
    '''
    '''=================================KD / FC=================================================================='''
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1",
                    help="請輸入第一個輸入檔案(.csv)")
    parser.add_argument("--input2",
                    help="請輸入第一個輸入檔案(.csv)")
    parser.add_argument("--output1",
                    help="請輸入第一個輸出檔案名(.csv)")
    parser.add_argument("--output2",
                    help="請輸入第二個輸出檔案名(.csv)")    
    parser.add_argument("--output3",
                    help="請輸入第合併輸出檔案名(.csv)")   
    args = parser.parse_args()
    fc = pd.read_csv(args.input1)
    kd = pd.read_csv(args.input2)
    select=['eosinophil count ','basophil count ','monocyte count ','eosinophil (%)','basophil (%)','monocyte (%)']
    fc_nan,kd_nan = count_na(select=select,fc=fc,kd=kd)
    print('fc:',f'{fc_nan}\n','kd:',f'{kd_nan}\n')
    fc_out ,kd_out = fill0_select(select=select,fc=fc,kd=kd)
    kdi = next(iter(kd_out[kd_out['eosinophil count ']==nan].index),'no match')
    fci = next(iter(fc[fc['eosinophil count ']==nan].index),'no match')
    #''' for drop ALT or AST
    kd_out = kd_out.drop(['AST/GOT','age (day)','age (year)'],axis = 1,inplace = False)
    fc_out = fc_out.drop(['AST/GOT','age (day)','age (year)'],axis = 1,inplace = False)
    #'''
    ## 註解 : 差一筆資料可能會源於是否有先採用性別欄位再去除掉空值(性別有一空值) 
    '''=======================主程式碼運行部分==================='''
    fc_out.insert(len(fc_out.columns),'Class', 0)
    kd_out.insert(len(kd_out.columns),'Class', 1)
    fc_out = fc_out.dropna() #這一步驟會將seg有缺值的病人刪除
    kd_out = kd_out.dropna()
    combine = pd.concat([fc_out,kd_out],axis = 0)#axis = 0時為縱向合併
    combine = combine.set_index('ID')

    fc_out.to_csv(args.output1)
    kd_out.to_csv(args.output2)
    combine.to_csv(args.output3)
    print('fc input file:',len(fc),'\tfc output file:',len(fc_out))
    print('kd input file:',len(kd),'\tkd output file:',len(kd_out))
    print('combine file:',len(combine))
   

