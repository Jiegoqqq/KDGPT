
from cmath import nan
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from IPython.display import display 
import os 


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
    fc = pd.read_csv('./zscore_table_ver2_output/fc_ver2.csv')
    kd = pd.read_csv('./zscore_table_ver2_output/kd_ver2.csv')
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
    '''data for 第一管血
    kd_out = kd_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit']]
    fc_out = fc_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit']]
    '''
    '''data for 第二管血
    kd_out = kd_out[['ID','age (month)','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ']]
    fc_out = fc_out[['ID','age (month)','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ']]
    '''
    '''data for 第三管血
    kd_out = kd_out[['ID','age (month)','AST/GOT','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','age (month)','AST/GOT','ALT/GPT','CRP']]
    '''
    '''data for 第三管血 NO ast
    kd_out = kd_out[['ID','age (month)','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','age (month)','ALT/GPT','CRP']]
    '''
    ''' data for 第一管+第二管
    kd_out = kd_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ']]
    fc_out = fc_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ']]
    '''
    ''' data for 第一管+第三管
    kd_out = kd_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','AST/GOT','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','AST/GOT','ALT/GPT','CRP']]
    '''
    ''' data for 第一管+第三管 no ast
    kd_out = kd_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','ALT/GPT','CRP']]
    '''
    ''' data for 第二管+第三管
    kd_out = kd_out[['ID','age (month)','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','AST/GOT','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','age (month)','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','AST/GOT','ALT/GPT','CRP']]
    '''
    ''' data for 第二管+第三管 no ast
    kd_out = kd_out[['ID','age (month)','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','age (month)','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','ALT/GPT','CRP']]
    '''
    ''' data for lam
    kd_out = kd_out[['ID','age (month)','Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Band (%)','Band count ','Seg (%)','Seg count ','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','age (month)','Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Band (%)','Band count ','Seg (%)','Seg count ','ALT/GPT','CRP']]
    '''
    '''data for 獨有 (no AST,monocyte count)
    kd_out = kd_out[['ID','RBC','Hematocrit','MCV','MCH','MCHC','basophil (%)','basophil count ','eosinophil count ','Lym count ','EOS z score','HB z score','PLT z score']]
    fc_out = fc_out[['ID','RBC','Hematocrit','MCV','MCH','MCHC','basophil (%)','basophil count ','eosinophil count ','Lym count ','EOS z score','HB z score','PLT z score']]
    '''
    '''data for Top 5 (already be the newest one)
    kd_out = kd_out[['ID','CRP','age (month)','EOS z score','ALT/GPT','monocyte (%)','性別']]
    fc_out = fc_out[['ID','CRP','age (month)','EOS z score','ALT/GPT','monocyte (%)','性別']]
    '''


    ''' for paper2 lings
    kd_out = kd_out[['ID','Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Seg (%)','ALT/GPT','CRP']]
    fc_out = fc_out[['ID','Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Seg (%)','ALT/GPT','CRP']]
    '''
    '''for paper2 our only
    kd_out = kd_out.drop(['Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Seg (%)','ALT/GPT','CRP','AST/GOT'],axis = 1,inplace = False)
    fc_out = fc_out.drop(['Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Seg (%)','ALT/GPT','CRP','AST/GOT'],axis = 1,inplace = False)
    '''
    '''for paper haung
    kd_out = kd_out.drop(['HB z score','Lym count ','eosinophil count ','monocyte count ','Seg count ','EOS z score','PLT z score','AST/GOT','MCV','Band count ','basophil count '],axis = 1,inplace = False)
    fc_out = fc_out.drop(['HB z score','Lym count ','eosinophil count ','monocyte count ','Seg count ','EOS z score','PLT z score','AST/GOT','MCV','Band count ','basophil count '],axis = 1,inplace = False)
    '''
    ## 註解 : 差一筆資料可能會源於是否有先採用性別欄位再去除掉空值(性別有一空值) 
    '''=======================主程式碼運行部分==================='''
    fc_out.insert(len(fc_out.columns),'Class', 0)
    kd_out.insert(len(kd_out.columns),'Class', 1)
    fc_out = fc_out.dropna() #這一步驟會將seg有缺值的病人刪除
    kd_out = kd_out.dropna()
    combine = pd.concat([fc_out,kd_out],axis = 0)#axis = 0時為縱向合併
    combine = combine.set_index('ID')

    folderpath = './data_preprocessing_combine_output/'
    if os.path.isdir(folderpath):
        pass
    else:
        os.makedirs(folderpath)
    fc_out.to_csv('./data_preprocessing_combine_output/fc_KeptGPT_foranova.csv')
    kd_out.to_csv('./data_preprocessing_combine_output/kd_KeptGPT_foranova.csv')
    combine.to_csv('./data_preprocessing_combine_output/combine_fc_kd_KeptGPT_foranova.csv')
    print('fc input file:',len(fc),'\tfc output file:',len(fc_out))
    print('kd input file:',len(kd),'\tkd output file:',len(kd_out))
    print('combine file:',len(combine))
   

