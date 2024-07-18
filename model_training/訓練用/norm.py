from asyncio.windows_events import NULL
import numpy as np
import pandas
import scipy
from scipy import stats as stats#用來分析資料
import matplotlib.pyplot as plt 
import pandas as pd  
import xlrd #讀取excel檔的模板

fc = pd.read_excel('../datas/fever-CBCWBC-datafor-成大.xlsx')
kd = pd.read_excel('../datas/KD-CBCWBC-datafor-成大.xlsx')
for parameter in fc:
    fc['{}'.format(parameter)].apply(lambda x:np.NaN if str(x).isspace() else x)
    kd['{}'.format(parameter)].apply(lambda x:np.NaN if str(x).isspace() else x)
fc = fc.dropna()
kd = kd.dropna()

normtest_result = pd.DataFrame()
i = 0


for parameter in fc:
    if parameter == 'ID':
        pass
    else:
        try:
            fc = fc.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
            fc['{}'.format(parameter)] = fc['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
            fc = fc.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float
            fc_parameter = fc['{}'.format(parameter)].describe()
            mean_fc = fc_parameter['mean']
            std_fc = fc_parameter['std']

            kd = kd.astype({'{}'.format(parameter):'str'})
            kd['{}'.format(parameter)] = kd['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)
            kd = kd.astype({'{}'.format(parameter):'float'})  
            kd_parameter = kd['{}'.format(parameter)].describe()
            mean_kd = kd_parameter['mean']
            std_kd = kd_parameter['std']

            fc_stat,fc_pvalue = stats.kstest(list(fc['{}'.format(parameter)]), 'norm',(mean_fc,std_fc))
            kd_stat,kd_pvalue = stats.kstest(list(kd['{}'.format(parameter)]) , 'norm',(mean_kd,std_kd))

            if float(fc_pvalue) < 0.001:
                sign = 1
            else:
                sign = 0

            if float(kd_pvalue) < 0.001:
                sign2 = 1
            else:
                sign2 = 0
            normtest_result.insert(i,column = '{}'.format(parameter), value = [float(fc_pvalue),int(sign),float(kd_pvalue),int(sign2)])
            i+=1
            norm_fc = stats.shapiro(list(fc['{}'.format(parameter)]))
            print(norm_fc)
        except:
            print('some error')
            pass

print(normtest_result)
normtest_result.to_csv('../datas/normtest.csv')
