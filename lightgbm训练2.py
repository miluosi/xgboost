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
Y = data.iloc[:,9]
Xtrain,Xtest,Ytrain,Ytest = TTS(X,Y,test_size=0.1,random_state=420)



#因为 LightGBM 使用的是 leaf-wise 的算法，因此在调节树的复杂程度时，使用的是 num_leaves 而不是 max_depth。大致换算关系：num_leaves = 2^(max_depth)，但是它的值的设置应该小于 2^(max_depth)
parameters = {
    'max_depth': [i for i in range(1,5)],
    'num_leaves': range(1,20),
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
#一个叶子上的最小数据量。
#一个叶子上的最小hessian和。默认设置为0.001，一般设置为1。不建议调整，增大数值会得到较浅的树深
parameters = {
'min_child_samples': range(1,20),
'min_child_weight':   [i/1000 for i in range(10)]
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 1,
                         num_leaves = 2,
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



parameters = {
    'feature_fraction': [i/10 for i in range(1,10)],
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 1,
                         num_leaves = 2,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=10,
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



parameters = {
     'bagging_fraction': [i/10 for i in range(1,11)],
     'bagging_freq': [i for i in range(1,11)],
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 1,
                         num_leaves = 2,
                         learning_rate = 0.1,
                         feature_fraction = 0.3,
                         min_child_samples=10,
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



parameters = {
     'cat_smooth': [i*5 for i in range(1,10)],
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 1,
                         num_leaves = 2,
                         learning_rate = 0.1,
                         feature_fraction = 0.3,
                         min_child_samples=10,
                         min_child_weight=0.0,
                         bagging_fraction = 0.9,
                         bagging_freq = 9,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 0,
                         num_iterations = 200,
                        )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='r2', cv=5)
gsearch.fit(Xtrain, Ytrain)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))



parameters = {
     'learning_rate': [i/10 for i in range(10)],
    'num_iterations': [(i+1)*100 for i in range(5) ]
}
gbm =          LGBR(objective = 'regression',
                         is_unbalance = True,
                         max_depth = 1,
                         num_leaves = 2,
                         learning_rate = 0.1,
                         feature_fraction = 0.3,
                         min_child_samples=10,
                         min_child_weight=0.0,
                         bagging_fraction = 0.9,
                         bagging_freq = 9,
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