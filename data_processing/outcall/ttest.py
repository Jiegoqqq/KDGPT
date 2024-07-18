# from asyncio.windows_events import NULL
from operator import index
import numpy as np
from scipy import stats as stats#用來分析資料
import matplotlib.pyplot as plt 
import pandas as pd  
import argparse




def ttest(input1,input2,path):
    ttest_result = pd.DataFrame()
    i = 0
    #print(fc.shape,kd.shape)# 印出資料維度
    #print(fc)
    #print(kd)
    #print(fc.info())#印出資料概況
    #print(kd.info())
    for parameter in fc:
        input1 = input1.astype({'{}'.format(parameter):'str'})#為了下一步，要先將所有資料轉成str
        input1['{}'.format(parameter)] = input1['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)#內部的非數值str都remove
        input1 = input1.astype({'{}'.format(parameter):'float'})#移除掉非數值str後將其轉回float
        input2 = input2.astype({'{}'.format(parameter):'str'})
        input2['{}'.format(parameter)] = input2['{}'.format(parameter)].str.extract('(-*\d+\.*\d*)',expand = False)
        input2 = input2.astype({'{}'.format(parameter):'float'})            
        try:
            fc_parameter = input1['{}'.format(parameter)].describe()
            kd_parameter = input2['{}'.format(parameter)].describe()
            #print(fc_parameter)
            #print('=================')
            #print(kd_parameter)

            mean_fc = float(fc_parameter['mean'])
            mean_kd = float(kd_parameter['mean'])

            std_fc = fc_parameter['std']
            std_kd = kd_parameter['std']

            nobs_fc = fc_parameter['count'] #觀察值
            nobs_kd = kd_parameter['count']

            modified_std1 = np.sqrt(np.float64(nobs_fc)/np.float64(nobs_fc-1)) * std_fc
            modified_std2 = np.sqrt(np.float64(nobs_kd)/np.float64(nobs_kd-1)) * std_kd
            (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean_fc, std1=modified_std1, nobs1=nobs_fc, mean2=mean_kd, std2=modified_std2, nobs2=nobs_kd)

            if float(pvalue) < 0.001:
                sign = 1
            else:
                sign = 0
            ttest_result.insert(i,column = '{}'.format(parameter), value = [float(pvalue),int(sign)])
            i+=1
        except Exception as e:
            print('=================')
            print(e,',{}'.format(parameter))
            print('=================')
            pass
    print(ttest_result)
    ttest_result.to_csv(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1",
                    help="請輸入第一個輸入檔案(.csv)")
    parser.add_argument("--input2",
                    help="請輸入第一個輸入檔案(.csv)")
    parser.add_argument("--output1",
                    help="請輸入第一個輸出檔案名(.csv)")
    args = parser.parse_args()
    fc = pd.read_csv(args.input1)
    kd = pd.read_csv(args.input2)
    path = args.output1
    ttest(fc,kd,path)




