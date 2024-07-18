
from model_training.訓練用.kfold import roc_plot,prc_plot,kfold_self,kfold_self_score,kfold_self_score_up,test_prc,test_roc,test_score,test_score_up
import lightgbm as lgb
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import warnings




#test_prc(model = model,file_train=file_train,file_test=file_test)
#test_roc(model = model,file_train=file_train,file_test=file_test,color_origin='orange',color_up = 'red')


'''

for i in range(4000,10000,1000):
    for j in range(4100,6000,2000):
        warnings.filterwarnings("ignore")
        model = LogisticRegression(random_state=i,penalty='none',C=4000,max_iter=300,solver='lbfgs',warm_start=False)
        path = 'dropna_ver1_dropall/k_fold_train_validation_20'
        print('i:',i,'j:',j)
        #path = 'data_old'
        loss = kfold_self(model = model,point_amount=300,space ='linspace',random_state=j,path = path,sub_plot=True,plot=False)
'''
#warnings.filterwarnings("ignore")
#model = lgb.LGBMClassifier(boosting = 'gbdt',learning_rate = 0.15,random_state = 42,min_data_in_leaf = 5,max_depth=10,num_leaves = 8)
model = xgb.XGBClassifier(random_state = 42,max_depth = 5,learning_rate =2,reg_lambda = 25000)#,gamma = 100)
#model = RandomForestClassifier(criterion = 'gini',max_depth =7,n_estimators = 100, min_samples_leaf= 10,max_features = 'log2',max_samples = 0.5,random_state=42)
#model = svm.SVC(kernel = 'rbf', C = 0.22, gamma =4.5e-2,max_iter = 3000,random_state = 42,tol =1e-3,probability=False)# (稍微還是有點overfitting)
#model = SVM(kernel='rbf',max_iter=4500,gamma = 2.5e-2,C=0.15)
#model = AdaBoostClassifier(n_estimators = 2500,random_state = 42,learning_rate = 0.05)
#model = LogisticRegression(random_state=42,penalty='none',C=4000,max_iter=7000,solver='lbfgs',warm_start=True)
#model  = MLPClassifier(random_state=42,learning_rate_init=0.01,max_iter=130,alpha = 0.2,solver = 'sgd')



'''for SHAP'''
#path = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/2-fold_train_z.csv'
#path_test = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/2-fold_test_z.csv'
#shapely(model=model,file_train=path,random_state_1=30,max_display=27,file_test=path_test)

''' for validation'''
#path = 'D:/CODE/KD/datas/our_only/k-fold/30' 
#train_loss,test_loss,train_loss_up,test_loss_up = kfold_self(model = model,point_amount=15,random_state=30,path = path,space='geomspace',loss_log=True,scoring='f1')
#prc_plot(path=path,random_state_2=30,model = model,model_amount=1,color_origin='orange',color_up='red',color_origin_2='blue',color_up_2='skyblue')
#score = kfold_self_score(path =path,model = model,scoring=['f1','precision','recall','specificity'],seperate = False,random_state=30)
#score.to_csv('D:/CODE/KD/codes/models_output/only2/lgbm/lgbm_spe_validation.csv')
#score_up = kfold_self_score_up(path = path,model = model,scoring=['f1','precision','recall','specificity'],seperate = False,random_state=30)
#score_up.to_csv('D:/CODE/KD/codes/models_output/only2/lgbm/lgbm_score_up_spe_validation.csv')


'''for test''' 
# file_test = 'D:/CODE/KD/datas/KeptGPT/k-fold/total-test_z.csv'
# file_train_2 ='D:/CODE/KD/datas/KeptGPT/k-fold/2-fold_train_z.csv'
# score = test_score(model = model,scoring=['f1','precision','recall','specificity'],random_state=30,seperate = False,space='linspace',file_test=file_test,file_train=file_train_2)
# score.to_csv('models_output/double model/logistic_score_spe_up_test.csv')

'''
model = LogisticRegression(random_state=40,penalty='none',C=4000,max_iter=300,solver='lbfgs',warm_start=True)
model_2 = AdaBoostClassifier(n_estimators = 2500,random_state = 42,learning_rate = 0.05)
#prc_plot(model = model,model_2=model_2,model_amount=2,color_origin='red',color_up='orange',color_origin_2='blue',color_up_2='skyblue')
roc_plot(model = model,model_2=model_2,model_amount=2,color_origin='red',color_up='orange',color_origin_2='blue',color_up_2='skyblue')
'''

#file_train = 'D:/CODE/KD/datas/KeptNo/k-fold/30/4-fold_train_z.csv'
#file_test = 'D:/CODE/KD/datas/KeptNo/k-fold/30/total-test_z.csv'
#model = AdaBoostClassifier(n_estimators = 3200,random_state = 42,learning_rate = 0.2)
#print(model)
#score = test_score(model = model,scoring=['f1','precision','recall','specificity'],random_state=30,seperate = False,file_test=file_test,file_train=file_train)
#score.to_csv('D:/CODE/KD/codes/models_output/KeptNo/Ada/ada_score_spe_test.csv')
#score_up= test_score_up(model = model,scoring=['f1','precision','recall','specificity'],random_state=30,seperate = False,file_test=file_test,file_train=file_train)
#score_up.to_csv('D:/CODE/KD/codes/models_output/KeptNo/Ada/ada_score_spe_up_test.csv')  
    
'======================test set======================================'

#file_train = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/3-fold_train_z.csv'
#file_test = 'D:/CODE/KD/datas/KeptGPT/k-fold/30/total-test_z.csv'
#file_train_2 ='dropna_ver1_dropall/k_fold_train_validation_30/5-fold_train_z.csv'
#model_2 = LogisticRegression(random_state=40,penalty='none',C=4000,max_iter=300,solver='lbfgs',warm_start=True)
#model = AdaBoostClassifier(n_estimators = 2500,random_state = 42,learning_rate = 0.05)
#score = test_score(model = model,scoring=['f1','precision','recall','specificity'],random_state=30,seperate = False,file_test=file_test,file_train=file_train)
#score.to_csv('D:/CODE/KD/codes/models_output/KeptGOTGPT/Ada/ada_score_spe_test.csv')
#score_up= test_score_up(model = model,scoring=['f1','precision','recall','specificity'],random_state=30,seperate = False,file_test=file_test,file_train=file_train)
#score_up.to_csv('D:/CODE/KD/codes/models_output/KeptGOTGPT/Ada/ada_score_spe_up_test.csv')
#test_prc(file_train = file_train,file_train_2=file_train_2,file_test=file_test,model=model,model_2=model_2,random_state_1=42,random_state_2 = 30,model_amount=1,color_origin='orange',color_up='lightsalmon',color_origin_2='blue',color_up_2='skyblue')



