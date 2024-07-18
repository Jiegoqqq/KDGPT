import pandas as pd 
import statsmodels.api as sm
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.graphics.regressionplots import abline_plot

def ols_model(data, model_type, feature,feature_list):
    data_use = data[data['model type'] == model_type]
    data_use.reset_index(inplace=True,drop=True)
    # Selecting relevant features from the DataFrame
    selected_features = feature[feature_list]
    selected_features['predict probability(as KD)'] = data_use['predict probability(as KD)']
    # Adding a constant column to the DataFrame
    selected_features = selected_features[selected_features['predict probability(as KD)']!='-']

    selected_features.reset_index(inplace=True,drop=True)
    X = sm.add_constant(selected_features[feature_list])
    Y = selected_features['predict probability(as KD)']
    fig, ax = plt.subplots(figsize=(10, 6))
    # predicted_values = model.predict(X.astype(float))
    # ax.scatter(selected_features['skin rash'], predicted_values, label='Predicted Y', color='red',alpha=1)
    ax.scatter(selected_features['前七項症狀加總總分'], Y.astype(float), label='Actual Y', alpha=0.5)
    plt.plot()
    ax.set_xlabel('skin rash')
    ax.set_ylabel('Y')
    ax.legend()
    plt.title('OLS Regression Results')
    # Show the plot
    plt.show()
    # model = sm.OLS(Y.astype(float), X.astype(float)).fit()
    # Identity matrix for F-test
    # r_matrix = np.identity(len(model.params))
    # with open(f'D:/CODE/KD/code_after_web/data/蔡小姐提供/KD症狀與血檢資料OLS/{model_type}.txt' ,'w') as f:
    #     f.write(f'case amount:{len(data_use)} used:{len(selected_features)}')
    #     f.write(f'\n{model.summary()}')
    #     f.write(f'\n{model.f_test(r_matrix)}')
    #     f.write(f'\nf value:{model.fvalue}')
    #     f.write(f'\nf pvalue:{model.f_pvalue}')
 
    # print(model.summary())
    # print(model.f_test(r_matrix))
    # print("f value:",model.fvalue)
    # print("f pvalue:",model.f_pvalue)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # predicted_values = model.predict(X.astype(float))
    # ax.scatter(selected_features['skin rash'], predicted_values, label='Predicted Y', color='red',alpha=1)
    # ax.scatter(selected_features['skin rash'], Y.astype(float), label='Actual Y', alpha=0.5)
    # plt.plot()
    # ax.set_xlabel('skin rash')
    # ax.set_ylabel('Y')
    # ax.legend()
    # plt.title('OLS Regression Results')
    # Show the plot
    # plt.show()
    return


if __name__ == "__main__":
    data = pd.read_csv("../KD症狀與血檢資料_結果.csv")
    feature = pd.read_csv("../KD症狀與血檢資料.csv")
    #['lip', 'eye', 'lymph>1.5', 'lymph腫脹', 'edematous ', 'skin rash', 'BCG']
    ols_model(data,'full model',feature,['前七項症狀加總總分'])

