from asyncio.windows_events import NULL
import numpy as np
from scipy import stats as stats#用來分析資料
import matplotlib.pyplot as plt 
import pandas as pd  


fc = pd.read_csv('D:/CODE/KD/code_after_web/data/origin/fc.csv')
kd = pd.read_csv('D:/CODE/KD/code_after_web/data/origin/kd.csv')
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
fc.to_csv('D:/CODE/KD/code_after_web/data/fc_drop.csv')
kd.to_csv('D:/CODE/KD/code_after_web/data/kd_drop.csv')
#將兩dataFrame合而為一
combine = pd.concat([fc,kd],axis = 0)#axis = 0時為縱向合併
combine.set_index('ID',inplace = True)
combine.to_csv('D:/CODE/KD/code_after_web/data/fc_kd_combine.csv')
print(fc.shape)
print(kd.shape)
print(combine.shape)
