from operator import index
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import os
folderpath = 'data_old'

if os.path.isdir(folderpath):
    pass
else:
    os.mkdir(folderpath)
for i in range(1,6):
    train_z = pd.read_csv(folderpath+'/{}-fold_train_z.csv'.format(i))
    test_z = pd.read_csv(folderpath+'/{}-fold_test_z.csv'.format(i))
    train_z = train_z.drop(columns = ['Segp','Lymp','eosinophilp','Bandp'])
    test_z = test_z.drop(columns = ['Segp','Lymp','eosinophilp','Bandp'])
    train_z.to_csv(folderpath+'/{}-fold_train_z.csv'.format(i))
    test_z.to_csv(folderpath+'/{}-fold_test_z.csv'.format(i))