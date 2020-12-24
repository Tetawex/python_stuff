from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils import check_array

import graphviz
import os

os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin/'  # Можно добавить graphviz руками
from matplotlib.pylab import rcParams
import xgboost as xgb

from IPython.display import display


# import math


def process_model(model, df, dependent, predictors):
    # Уравнение регрессии
    equation = dependent + ' ~ ' + (' + '.join(predictors))

    y, X = dmatrices(equation, data=df, return_type='dataframe')

    renamer = lambda x: x.replace('[', '_').replace(']', '')

    y = y.rename(columns=renamer)
    X = X.rename(columns=renamer)

    # Делим выборку
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    display(y_pred)

    # Считаем метрики качества
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    adjusted_r_2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    print('MSE: %.3f' % mse)
    print('Mean Error: %.3f' % rmse)
    print('R^2: %.3f' % r2)
    print('Adjusted R^2: %.3f' % adjusted_r_2)

    print(type(model))

    y_test = y_test.apply(np.exp)
    y_pred = np.exp(y_pred.flatten().reshape(-1, 1))

    print('MAPE: %.3f' % mean_absolute_percentage_error(y_test, y_pred))
    # Строим график
    y_pred_scatter_index = list(range(0, len(y_test)))

    vec_test = pd.Series(y_test[dependent], index=y_pred_scatter_index, dtype='float')
    vec_pred = pd.Series(y_pred.flatten(), index=y_pred_scatter_index, dtype='float')

    frame = {'Actual Price, $': vec_test, 'Predicted Price, $': vec_pred, }

    comp_df = pd.DataFrame(frame, index=y_pred_scatter_index) \
        # .join(df['property_type_computed'].reindex(y_pred_scatter_index), how='inner')

    y_min_val = min(vec_pred.min(), vec_test.min())
    y_max_val = max(vec_pred.max(), vec_test.max())

    plt.figure(figsize=(8, 8))
    plt.axline((y_min_val, y_min_val), (y_max_val, y_max_val), color='black')
    # plt.axline((1.2, 1.2), (1.8, 1.8), color='black')
    sns.scatterplot(x='Actual Price, $',
                    y='Predicted Price, $',
                    #                 hue='property_type_computed',
                    data=comp_df).set_title('Prediction quality')


    return model

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def process_cl_model(model, df, dependent, cat_labels, predictors, print_tree):
    # Уравнение
    equation = dependent + ' ~ ' + (' + '.join(predictors))

    y, X = dmatrices(equation, data=df, return_type='dataframe')

    renamer = lambda x: x.replace('[', '_').replace(']', '')

    y = y.rename(columns=renamer)
    X = X.rename(columns=renamer)

    # Делим выборку
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Считаем метрики качества
    target_names = list(map(lambda s: 'Class ' + s, df[cat_labels].unique().astype('str').tolist()))
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True)).iloc[:-1, :].T,
        annot=True,
        vmin=0.4, vmax=1)
    plt.show()
    display(classification_report(y_test, y_pred, target_names=target_names, output_dict=True))
    # display(pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True)))

    # Визуализация

    if print_tree:
        ft_weights_xgb_reg = pd.DataFrame(model.feature_importances_, columns=['weight'], index=X_train.columns)
        ft_weights_xgb_reg.sort_values('weight', ascending=False, inplace=True)
        display(ft_weights_xgb_reg)

        viz_tree(model)


def viz_tree(model):
    rcParams['figure.figsize'] = 80, 120

    # xgb.plot_tree(xg_reg, rankdir='LR'); plt.show()
    xgb.plot_tree(model, num_trees=0, rankdir='LR');
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    fig.savefig('tree.png')
    plt.show(fig)

# def view_summary(model, ):
#
#
# def view_metrics(model, )
