# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
#
# TARGET_COLUMN = 'price'
#
# train, test = train_test_split(new_df, test_size=0.2)
#
# predictors = train[train.columns[train.columns != TARGET_COLUMN]].fillna(train.mean())
# target = train[[TARGET_COLUMN]].fillna(train.mean())
#
# test_predictors = test[test.columns[test.columns != TARGET_COLUMN]].fillna(test.mean())
# test_target = test[[TARGET_COLUMN]].fillna(test.mean())
#
# # Create linear regression object
# regr = linear_model.LinearRegression()
#
# # Train the model using the training sets
# regr.fit(predictors, target)
#
# # Make predictions using the testing set
# prediction = regr.predict(test_predictors)
#
# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print('Mean squared error: %.2f'
#       % mean_squared_error(test_target, prediction))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(test_target, prediction))
# Модель 1 - деревья решений
tree_model = model_utils.process_model(
    model=xgb.XGBRegressor(objective='reg:squarederror',
                           n_jobs=8,
                           tree_method='gpu_hist',
                           colsample_bytree=0.4,
                           learning_rate=0.01,
                           max_depth=9,
                           min_child_weight=4,
                           alpha=0,
                           gamma=0,
                           n_estimators=1000,
                           seed=123),
    df=final_df,
    dependent=TARGET_COLUMN,
    predictors=PREDICTORS)
