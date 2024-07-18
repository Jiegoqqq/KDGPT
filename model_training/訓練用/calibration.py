import warnings
import json
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,recall_score,precision_recall_curve,auc,precision_score,f1_score
from sklearn.utils import resample,shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from learn_curving_fold import plot_learning_curves
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import copy
#from svm-gpu import SVM
from IPython.display import display
import os
import joblib
from tqdm import trange
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
warnings.filterwarnings("ignore")



fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
file_test_full = "D:/CODE/KD/code_after_web/data/Top5_under7(ByKeptGPT,with monocyte count)/k-fold/30/total-test_z-fold4.csv"
fold_test_full = pd.read_csv(file_test_full)
try:
    fold_test_full = fold_test_full.drop(columns=['Unnamed: 0'])
except:
    pass
fold_test_full = fold_test_full.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
one_message_test = fold_test_full[fold_test_full["Class"] == 1 ]
zero_message_test =fold_test_full[fold_test_full["Class"] == 0]
zero_message_down_test = resample(zero_message_test,replace=True,n_samples=len(one_message_test),random_state=42)
fold_test_up = pd.concat([one_message_test,zero_message_down_test])
print(len(one_message_test),len(zero_message_down_test),len(fold_test_up))
fold_test_up = shuffle(fold_test_up,random_state = 30)
fold_test_up= fold_test_up.reset_index(drop = True)
## set the train set for single fold
fold_test_up= fold_test_up.reset_index(drop = True)
x_test_full = fold_test_up.drop(columns=["Class"])
y_test_full = fold_test_up['Class']
model_full = joblib.load("D:/CODE/KD/code_after_web/output/Under 7/Top5_under7(ByKeptGPT,with monocyte count)/lgbm/4-model")
display = CalibrationDisplay.from_estimator(model_full,x_test_full,y_test_full,n_bins=10,name='lgbm',ax=ax_calibration_curve,)
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
ax = fig.add_subplot(gs[2, 0])
ax.hist(display.y_prob,range=(0, 1),bins=10,label='lgbm',)
ax.set(title='lgbm', xlabel="Mean predicted probability", ylabel="Count")
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (lgbm)")
'==============================upsampling train/origin======================='
model =  lgb.LGBMClassifier(boosting = 'gbdt',learning_rate = 0.2,random_state = 42,min_data_in_leaf = 10,max_depth = 5,num_leaves = 5)
svc_isotonic = CalibratedClassifierCV(model, method="isotonic")
svc_sigmoid = CalibratedClassifierCV(model, method="sigmoid")
fold_train = pd.read_csv('D:/CODE/KD/code_after_web/data/Top5_under7(ByKeptGPT,with monocyte count)/k-fold/30/4-fold_train_z.csv')
## rename(消去columns項內非英文以及數字的解)
try:
    fold_train = fold_train.drop(columns=['Unnamed: 0'])
except:
    pass
fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
## upsampling train set 
one_message = fold_train[fold_train["Class"] == 1 ]
zero_message =fold_train[fold_train["Class"] == 0]
one_message_up = resample(one_message,
            replace=True,
            n_samples=len(zero_message),
            random_state=42)
fold_train = pd.concat([one_message_up,zero_message])
fold_train = shuffle(fold_train,random_state = 30)
fold_train= fold_train.reset_index(drop = True)
## set the train set for single fold
x_train = fold_train.drop(columns=['Class'])
y_train = fold_train['Class']

## get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
roc_model = svc_isotonic.fit(x_train,y_train)

display = CalibrationDisplay.from_estimator(
    roc_model,
    x_test_full,
    y_test_full,
    n_bins=10,
    name='lgbm+isotonic',
    ax=ax_calibration_curve,
)
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
ax = fig.add_subplot(gs[2, 1])
ax.hist(
        display.y_prob,
        range=(0, 1),
        bins=10,
        label='lgbm+isotonic',
    )
ax.set(title='lgbm+isotonic', xlabel="Mean predicted probability", ylabel="Count")

'==============================upsampling train/origin======================='
model =  lgb.LGBMClassifier(boosting = 'gbdt',learning_rate = 0.2,random_state = 42,min_data_in_leaf = 10,max_depth = 5,num_leaves = 5)
svc_isotonic = CalibratedClassifierCV(model, method="isotonic")
svc_sigmoid = CalibratedClassifierCV(model, method="sigmoid")
fold_train = pd.read_csv('D:/CODE/KD/code_after_web/data/Top5_under7(ByKeptGPT,with monocyte count)/k-fold/30/4-fold_train_z.csv')
## rename(消去columns項內非英文以及數字的解)
try:
    fold_train = fold_train.drop(columns=['Unnamed: 0'])
except:
    pass
fold_train = fold_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
## upsampling train set 
one_message = fold_train[fold_train["Class"] == 1 ]
zero_message =fold_train[fold_train["Class"] == 0]
one_message_up = resample(one_message,
            replace=True,
            n_samples=len(zero_message),
            random_state=42)
fold_train = pd.concat([one_message_up,zero_message])
fold_train = shuffle(fold_train,random_state = 30)
fold_train= fold_train.reset_index(drop = True)
## set the train set for single fold
x_train = fold_train.drop(columns=['Class'])
y_train = fold_train['Class']

## get the tn,fp,fn,tp by confusion and therefore we can compute the specificity
roc_model = svc_sigmoid.fit(x_train,y_train)

display = CalibrationDisplay.from_estimator(
    roc_model,
    x_test_full,
    y_test_full,
    n_bins=10,
    name='lgbm+sigmoid',
    ax=ax_calibration_curve,
)
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
ax = fig.add_subplot(gs[3, 0])
ax.hist(
        display.y_prob,
        range=(0, 1),
        bins=10,
        label='lgbm+sigmoid',
    )
ax.set(title='lgbm+sigmoid', xlabel="Mean predicted probability", ylabel="Count")
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (lgbm)")
plt.tight_layout()
plt.show()