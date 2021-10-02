import pandas as pd
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor as LGBR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data = pd.read_excel(r'C:\Users\86139\Desktop\B\原始数据.xlsx')


X = data.iloc[:,1:7]
Y = data.iloc[:,7]
Xtrain,Xtest,Ytrain,Ytest = TTS(X,Y,test_size=0.1,random_state=420)



#Step 1 调整max_depth 和 num_leaves
parameters = {
    'max_depth': [2,4,6,8],
    'num_leaves': range(11,30),
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 6,
                         num_leaves = 40,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=21,
                         min_child_weight=0.001,
                         bagging_fraction = 1,
                         bagging_freq = 2,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 0,
                         num_iterations = 200,
                        )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='r2', cv=5)
gsearch.fit(Xtrain, Ytrain)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

#Step2 调整min_data_in_leaf 和 min_sum_hessian_in_leaf
parameters = {
'min_child_samples': range(15,20),
'min_child_weight':   [i/1000 for i in range(10)]
}

gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 2,
                         num_leaves = 11,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=21,
                         min_child_weight=0.001,
                         bagging_fraction = 1,
                         bagging_freq = 2,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 0,
                         num_iterations = 200,
                        )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='r2', cv=5)
gsearch.fit(Xtrain, Ytrain)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

#Step2 调整min_data_in_leaf 和 min_sum_hessian_in_leaf
parameters = {
    'feature_fraction': [i/10 for i in range(1,10)],
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 2,
                         num_leaves = 11,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=15,
                         min_child_weight=0.0,
                         bagging_fraction = 1,
                         bagging_freq = 2,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 0,
                         num_iterations = 200,
                        )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='r2', cv=5)
gsearch.fit(Xtrain, Ytrain)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

#Step2 调整min_data_in_leaf 和 min_sum_hessian_in_leaf
parameters = {
     'bagging_fraction': [i/10 for i in range(1,11)],
     'bagging_freq': [i for i in range(1,11)],
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 2,
                         num_leaves = 11,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=15,
                         min_child_weight=0.0,
                         bagging_fraction = 1,
                         bagging_freq = 2,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 0,
                         num_iterations = 200,
                        )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='r2', cv=5)
gsearch.fit(Xtrain, Ytrain)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

#Step2 调整min_data_in_leaf 和 min_sum_hessian_in_leaf
parameters = {
     'cat_smooth': [i*5 for i in range(1,10)],
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 2,
                         num_leaves = 11,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=15,
                         min_child_weight=0.0,
                         bagging_fraction = 0.9,
                         bagging_freq = 5,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 0,
                         num_iterations = 200,
                        )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='r2', cv=5)
gsearch.fit(Xtrain, Ytrain)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

#Step 7 最后，本人会适当调小learning_rate的值以及调整num_iterations的大小。
parameters = {
     'learning_rate': [i/10 for i in range(10)],
    'num_iterations': [(i+1)*100 for i in range(10) ]
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 2,
                         num_leaves = 11,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=15,
                         min_child_weight=0.0,
                         bagging_fraction = 0.9,
                         bagging_freq = 5,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 5,
                         num_iterations = 200,
                        )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='r2', cv=5)
gsearch.fit(Xtrain, Ytrain)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))


reg = gsearch.best_estimator_
plt.figure(figsize=(20, 10), dpi=100)
plt.bar(Xtrain.columns, reg.feature_importances_)
plt.xticks(range(Xtrain.shape[1]))
plt.ylim(0,0.6)
plt.xlabel("feature name")
plt.ylabel("feature importance")
plt.title("温度与不同催化剂对对C4烯烃选择性(%)影响的重要性")
plt.rcParams['figure.figsize'] = (20.0, 8.0)
print(reg)
print("R^2",reg.score(Xtest,Ytest))#R^2评估指标
print("MSE",MSE(Ytest,reg.predict(Xtest)))
print("importance",reg.feature_importances_)