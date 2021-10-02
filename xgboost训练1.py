from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
import pandas as pd
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from numpy import nan as NA
import pickle



data = pd.read_excel(r'C:\Users\86139\Desktop\B\原始数据.xlsx')
X = data.iloc[:,1:7]
Y = data.iloc[:,7]
Xtrain,Xtest,Ytrain,Ytest = TTS(X,Y,test_size=0.1,random_state=420)



#第二步：调整树结构
#和GBM中的参数相同，这个值为树的最大深度。
#决定最小叶子节点样本权重和，这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(2,7,2)
}
gsearch1 = GridSearchCV(estimator =XGBR( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear',
 nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='r2',n_jobs=4, cv=5)
gsearch1.fit(Xtrain,Ytrain)
gsearch1.best_params_, gsearch1.best_score_


#在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
#这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的
param_test3 = {
 'gamma':[i/100.0 for i in range(0,100)]
}
gsearch1 = GridSearchCV(estimator =XGBR( learning_rate =0.1, n_estimators=140, max_depth=3,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear',
 nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test3, scoring='r2',n_jobs=4, cv=5)
gsearch1.fit(Xtrain,Ytrain)
gsearch1.best_params_, gsearch1.best_score_


#这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
#colsample_bytree[默认1]和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。
param_test4 = {
 'subsample':[i/10.0 for i in range(1,10)],
 'colsample_bytree':[i/10.0 for i in range(1,10)]
}
gsearch4 = GridSearchCV(estimator =XGBR( learning_rate =0.1, n_estimators=140, max_depth=3,
 min_child_weight=2, gamma=0.8, subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear',
 nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test4, scoring='r2',n_jobs=4, cv=5)
gsearch4.fit(Xtrain,Ytrain)
gsearch4.best_params_, gsearch4.best_score_


#第五步 调整正则项
param_test5 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch5 = GridSearchCV(estimator = XGBR( learning_rate =0.1, n_estimators=140, max_depth=3,
 min_child_weight=2, gamma=0, subsample=0.6, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test5, scoring='r2',n_jobs=4, cv=5)
gsearch5.fit(Xtrain,Ytrain)
gsearch5.best_params_, gsearch5.best_score_


#第六步 调整学习速率
param_test6 = {
 'learning_rate':[0, 0.001, 0.005, 0.01, 0.05,0.1,0.5,1]
}
gsearch5 = GridSearchCV(estimator = XGBR( learning_rate =0.1, n_estimators=140, max_depth=3,
 min_child_weight=2, gamma=0, subsample=0.6, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test6, scoring='r2',n_jobs=4, cv=5)
gsearch5.fit(Xtrain,Ytrain)
gsearch5.best_params_, gsearch5.best_score_



reg = gsearch5.best_estimator_
plt.figure(figsize=(20, 10), dpi=100)
plt.bar(Xtrain.columns, reg.feature_importances_)
plt.xticks(range(Xtrain.shape[1]))
plt.ylim(0,0.6)
plt.xlabel("feature name")
plt.ylabel("feature importance")
plt.title("温度与不同催化剂对乙醇转化率(%)影响的重要性")
plt.rcParams['figure.figsize'] = (20.0, 8.0)
print(reg)
print("R^2",reg.score(Xtest,Ytest))#R^2评估指标
print("MSE",MSE(Ytest,reg.predict(Xtest)))
print("importance",reg.feature_importances_)