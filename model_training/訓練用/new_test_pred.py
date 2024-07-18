import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

def predict(label_col,class_list,test_file,origin_file,model_type):
    test_file = test_file[test_file['model type'] == model_type].reset_index(drop = True)
    origin_file[['predict result','predict probability']] = test_file[['predict result','predict probability']]
    TP = len(origin_file[origin_file['predict result'] =="KD"][origin_file[label_col] == class_list[1]])
    FP = len(origin_file[origin_file['predict result'] =="KD"][origin_file[label_col] == class_list[0]])
    TN = len(origin_file[origin_file['predict result'] =="Non KD"][origin_file[label_col] == class_list[0]])
    FN = len(origin_file[origin_file['predict result'] =="Non KD"][origin_file[label_col] == class_list[1]])
    print(TP,FP,TN,FN,TP+FP+TN+FN)
    acc = (TP+TN)/(TP+FP+TN+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    specificity = TN/(TN+FP)
    PPV = (TP)/(TP+FP)
    NPV = (TN)/(FN+TN) 
    FPR = (FP)/(TN+FP)
    TPR = (TP)/(TP+FN)
    FNR = (FN)/(TP+FN)
    TNR = (TN)/(TN+FP)
    P_LR = TPR/FPR
    N_LR = FNR/TNR
    print(f"case be used by model:{len(origin_file[origin_file['predict result']!='-'])}, total case:{len(origin_file)}")
    print(f"real KD cases:{TP+FN}, real FC cases:{FP+TN}")
    return({'total cases':[len(origin_file)],
            'model uses cases':[len(origin_file[origin_file['predict result']!='-'])],
            'real KD in model cases':[TP+FN],
            'real FC in model cases':[FP+TN],
            "true positive" :[TP],
            "false positive":[FP],
            "true negative":[TN],
            "false negative":[FN],
            'recall':[recall],
            'precision':[precision],
            'specificity':[specificity],
            'accuracy':[acc],'PPV':[PPV],
            'NPV':[NPV],'P_LR':[P_LR],
            'N_LR':[N_LR]
            })


if __name__ =='__main__':
    test_file = pd.read_csv("D:/CODE/KD/code_after_web/data/蔡小姐提供/2024-CBC-DC-CRP-空白表格邀請合作-荆州市第一人民医院潘炎_output.csv")
    origin_file = pd.read_excel("D:/CODE/KD/code_after_web/data/蔡小姐提供/2024-CBC-DC-CRP-空白表格邀請合作-荆州市第一人民医院潘炎.xlsx")
    label_col = '編號 KD: 1, FC: 2'
    class_list = [2,1] ## 前者為FC class,後者為KD class
    model_type='Top5'
    performance = predict(label_col,class_list,test_file,origin_file,model_type)
    performance_df = pd.DataFrame.from_dict(performance)
    print(performance_df)   
    performance_df.to_csv(f"D:/CODE/KD/sample_web/2024-CBC-DC-CRP-空白表格邀請合作-荆州市第一人民医院潘炎_{model_type}成效.csv",index=False)
    

    ## count the case that can't fit any model 
    # index_list = []
    # count = 0
    # for i in range(len(origin_file)):
    #     test_file_case = test_file[test_file['index'] == i+1]
    #     if len(test_file_case[test_file_case['predict result']!='-']) == 0:
    #         count += 1
    #         index_list.append(i+2)
    # print('\nindex in origin_file:',index_list,'\n\n cases count for not fitting any model:',count)