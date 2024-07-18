import pandas as pd 
from tqdm import trange
import os 
import re 

def FilterByCol(kfold_data_path,output_path,kept): # This will filter out the column that we need 
    kept = list(map(lambda x:re.sub('[^A-Za-z0-9_]+', 'p', x),kept))
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)
    for i in trange(1,6):
        train = pd.read_csv(f"{kfold_data_path}/{i}-fold_train.csv")
        validation = pd.read_csv(f"{kfold_data_path}/{i}-fold_test.csv")
        train = train[kept]
        validation = validation[kept]
        train.to_csv(f"{output_path}/{i}-fold_train.csv",index=False)
        validation.to_csv(f"{output_path}/{i}-fold_test.csv",index=False)
    test = pd.read_csv(f"{kfold_data_path}/total-test.csv")
    test = test[kept]
    test.to_csv(f"{output_path}/total-test.csv",index=False)

def FilterByAge(kfold_data_path,output_path,cut_off): ## This will filter out the patience that under the cutoff age 
    col = list(map(lambda x:re.sub('[^A-Za-z0-9_]+', 'p', x),['age month()']))
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)
    for i in trange(1,6):
        train = pd.read_csv(f"{kfold_data_path}/{i}-fold_train.csv")
        validation = pd.read_csv(f"{kfold_data_path}/{i}-fold_test.csv")
        train = train[train[col[0]]/12<cut_off]
        validation = validation[validation[col[0]]/12<cut_off]
        train.to_csv(f"{output_path}/{i}-fold_train.csv",index=False)
        validation.to_csv(f"{output_path}/{i}-fold_test.csv",index=False)
    test = pd.read_csv(f"{kfold_data_path}/total-test.csv")
    test = test[test[col[0]]/12<cut_off]
    test.to_csv(f"{output_path}/total-test.csv",index=False)
    
if __name__=='__main__':
    output_path = './haung_kfold_output'
    # output_path = '/home/cosbi/KDGPT/data_processing/under7_kfold_output'
    kfold_data_path = './under7_kfold_output'

    haung = ['age (month)','WBC (1000)','Hemoglobin','Hematocrit','MCH','RBC','MCHC','Platelets','CRP','ALT/GPT','Lym (%)','Seg (%)','eosinophil (%)','basophil (%)','Band (%)','monocyte (%)','Class']
    lam = ['age (month)','Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Band (%)','Band count ','Seg (%)','Seg count ','ALT/GPT','CRP','Class']
    ling=['Platelets','HB z score','WBC (1000)','Lym (%)','eosinophil (%)','monocyte (%)','Seg (%)','ALT/GPT','CRP','Class']
    Top5 = ['CRP','PLT z score','EOS z score','ALT/GPT','monocyte (%)','Class']
    first_blood = ['age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','Class']
    first_plus_two = ['age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','Class']
    first_plus_two_plus_ALT = ['age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','ALT/GPT','Class']
    first_plus_two_plus_CRP = ['age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','CRP','Class']
    without_z = ['age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','MCV','MCHC','Hematocrit','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','CRP','ALT/GPT','Class']
    without_count = ['age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','PLT z score','HB z score','MCV','MCHC','Hematocrit','Lym (%)','Seg (%)','eosinophil (%)','EOS z score','basophil (%)','Band (%)','monocyte (%)','CRP','ALT/GPT','Class']
    withou_count_z = ['age (month)','Platelets','Hemoglobin','WBC (1000)','MCH','RBC','MCV','MCHC','Hematocrit','Lym (%)','Seg (%)','eosinophil (%)','basophil (%)','Band (%)','monocyte (%)','CRP','ALT/GPT','Class']
    two = ['age (month)','Lym (%)','Lym count ','Seg (%)','Seg count ','eosinophil (%)','eosinophil count ','EOS z score','basophil (%)','basophil count ','Band (%)','Band count ','monocyte (%)','monocyte count ','Class']

    FilterByCol(kfold_data_path,output_path,haung)
    # FilterByAge(kfold_data_path,output_path,7)