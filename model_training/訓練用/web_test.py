import pandas as pd 
data = pd.read_csv("D:/CODE/KD/KD症狀與血檢資料.csv")
cols_crp = ['WBC(1000)', 'RBC', 'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'Platelets', 'Seg (%)', 'Lym (%)', 'monocyte (%)', 'eosinophil (%)', 'basophil (%)', 'Band (%)', 'CRP']
input_df_crp = data[cols_crp]
print(input_df_crp)