# Sebastian Raschka 2014-2022
# mlxtend Machine Learning Library Extensions
#
# A function for plotting learning curves of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange,tqdm

def plot_learning_curves(
    X_train,
    y_train,
    X_test,
    y_test,
    clf,
    train_marker="o",
    test_marker="^",
    scoring="misclassification error",
    suppress_plot=False,
    print_model=True,
    title_fontsize=12,
    style="default",
    legend_loc="best",
    point_amount = 20,
    space = 'geomspace'
):
    """Plots learning curves of a classifier.

    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
        Feature matrix of the training dataset.
    y_train : array-like, shape = [n_samples]
        True class labels of the training dataset.
    X_test : array-like, shape = [n_samples, n_features]
        Feature matrix of the test dataset.
    y_test : array-like, shape = [n_samples]
        True class labels of the test dataset.
    clf : Classifier object. Must have a .predict .fit method.
    train_marker : str (default: 'o')
        Marker for the training set line plot.
    test_marker : str (default: '^')
        Marker for the test set line plot.
    scoring : str (default: 'misclassification error')
        If not 'misclassification error', accepts the following metrics
        (from scikit-learn):
        {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
        'f1_weighted', 'f1_samples', 'log_loss',
        'precision', 'recall', 'roc_auc',
        'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
        'median_absolute_error', 'r2'}
    suppress_plot=False : bool (default: False)
        Suppress matplotlib plots if True. Recommended
        for testing purposes.
    print_model : bool (default: True)
        Print model parameters in plot title if True.
    title_fontsize : int (default: 12)
        Determines the size of the plot title font.
    style : str (default: 'default')
        Matplotlib style. For more styles, please see
        https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    legend_loc : str (default: 'best')
        Where to place the plot legend:
        {'best', 'upper left', 'upper right', 'lower left', 'lower right'}

    Returns
    ---------
    errors : (training_error, test_error): tuple of lists

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_learning_curves/

    """
    if scoring != "misclassification error":
        from sklearn import metrics

        scoring_func = {
            "accuracy": metrics.accuracy_score,
            "average_precision": metrics.average_precision_score,
            "f1": metrics.f1_score,
            "f1_micro": metrics.f1_score,
            "f1_macro": metrics.f1_score,
            "f1_weighted": metrics.f1_score,
            "f1_samples": metrics.f1_score,
            "log_loss": metrics.log_loss,
            "precision": metrics.precision_score,
            "recall": metrics.recall_score,
            "roc_auc": metrics.roc_auc_score,
            "adjusted_rand_score": metrics.adjusted_rand_score,
            "mean_absolute_error": metrics.mean_absolute_error,
            "mean_squared_error": metrics.mean_squared_error,
            "median_absolute_error": metrics.median_absolute_error,
            "r2": metrics.r2_score,
        }

        if scoring not in scoring_func.keys():
            raise AttributeError("scoring must be in", scoring_func.keys())

    else:

        def misclf_err(y_predict, y):
            return (y_predict != y).sum() / float(len(y))

        scoring_func = {"misclassification error": misclf_err}

    training_errors = []
    test_errors = []
    if (space =='geomspace'):
        rng = [int(i) for i in np.geomspace(5, X_train.shape[0], point_amount)][1:]
    elif(space =='linspace'):
        rng = [int(i) for i in np.linspace(5, X_train.shape[0], point_amount)][1:] #沒有取定0個位置，所以rng會有amount-1個值
        
    #progress = tqdm(total=len(rng))
    for r in rng:
        #print('log:',r+1)
        model = clf.fit(X_train[:r], y_train[:r])

        y_train_predict = clf.predict(X_train[:r])
        y_test_predict = clf.predict(X_test)
        if y_train_predict.ndim != 1:
            y_train_predict = y_train_predict.reshape(y_train_predict.shape[0],)
        if y_test_predict.ndim != 1:
            y_test_predict = y_test_predict.reshape(y_test_predict.shape[0],)
        
        train_misclf = scoring_func[scoring](y_train[:r], y_train_predict)
        training_errors.append(train_misclf)

        test_misclf = scoring_func[scoring](y_test, y_test_predict)
        test_errors.append(test_misclf)
        #progress.update(1)
    errors = (training_errors, test_errors)
    return errors

'''
    if not suppress_plot:
        with plt.style.context(style):
            plt.plot(
                np.arange(10, 101, 10),
                training_errors,
                label="training set",
                marker=train_marker,
            )
            plt.plot(
                np.arange(10, 101, 10),
                test_errors,
                label="test set",
                marker=test_marker,
            )
            plt.xlabel("Training set size in percent")

    if not suppress_plot:
        with plt.style.context(style):
            plt.ylabel("Performance ({})".format(scoring))
            if print_model:
                plt.title(
                    "Learning Curves\n\n{}\n".format(model), fontsize=title_fontsize
                )
            plt.legend(loc=legend_loc, numpoints=1)
            plt.xlim([0, 110])
            max_y = max(max(test_errors), max(training_errors))
            min_y = min(min(test_errors), min(training_errors))
            plt.ylim([min_y - min_y * 0.15, max_y + max_y * 0.15])
    '''

