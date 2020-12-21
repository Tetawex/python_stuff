xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
                          n_jobs = 8,
                          tree_method='gpu_hist',
                          colsample_bytree = 0.9,
                          learning_rate = 0.01,
                          max_depth = 7,
                          min_child_weight = 4,
                          alpha = 0.4,
                          gamma = 0,
                          n_estimators = 1000,
                          seed=123)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)