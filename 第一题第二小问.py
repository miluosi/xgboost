import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


import seaborn as sns
import pandas as pd
sns.set(font_scale = 1)
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(10, 10), dpi=100)
data1 = pd.read_excel(r'C:\Users\86139\Desktop\B\1.2.xlsx')
a=data1.iloc[:,0:].corr()
sns.heatmap(a, annot=True, vmax=1, square=True, cmap="Blues")
plt.title("相关系数矩阵图")
plt.show()



data = pd.read_excel(r'C:\Users\86139\Desktop\B\1.2.xlsx')
#生成多项式回归的函数，默认参数为2
def PolynomialRegression(degree=2,**kwarges):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression())


from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression
time = data['时间']
al = data['乙醇转化率(%)']
fig = plt.figure(figsize=(15, 8), dpi=100)
score =[]
coef = []
for i in range(0,4):
    i  =i+1
    poly = PF(degree=i)
    X_ = poly.fit_transform(np.array(time).reshape(-1, 1))
    # 训练数据的拟合
    LinearR_ = LinearRegression().fit(X_, al)
    X_curve = np.linspace(0,300,500).reshape(-1,1)
    X_curve1 = PF(degree=i).fit_transform(X_curve)
    plt.scatter(time,al,color='k')
    y_curve = LinearR_.predict(X_curve1)
    plt.plot(X_curve,y_curve,label='degree{a}'.format(a = i))
    plt.title("乙醇转化率(%)随时间变化拟合图像")
    plt.xlabel("时间")
    plt.ylabel("乙醇转化率(%)")
    coef.append(LinearR_.coef_)
    score.append(LinearR_.score(X_,al))
    plt.legend()
    plt.show()
a = range(1,10)
print([*zip(a,coef,score)])


from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression
fig = plt.figure(figsize=(15, 8), dpi=100)
al = data['乙烯选择性']
score =[]
coef = []
for i in range(0,4):
    i  =i+1
    poly = PF(degree=i)
    X_ = poly.fit_transform(np.array(time).reshape(-1, 1))
    # 训练数据的拟合
    LinearR_ = LinearRegression().fit(X_, al)
    X_curve = np.linspace(0,300,500).reshape(-1,1)
    X_curve1 = PF(degree=i).fit_transform(X_curve)
    plt.scatter(time,al,color='k')
    y_curve = LinearR_.predict(X_curve1)
    plt.plot(X_curve,y_curve,label='degree{a}'.format(a = i))
    coef.append(LinearR_.coef_)
    score.append(LinearR_.score(X_,al))
    plt.title("乙烯选择性随时间变化拟合图像")
    plt.xlabel("时间")
    plt.ylabel("乙烯选择性")
    plt.legend()
    plt.show()
a = range(1,10)
print([*zip(a,coef,score)])



fig = plt.figure(figsize=(15, 8), dpi=100)
al = data.iloc[:,5]
score =[]
coef = []
for i in range(0,4):
    i  =i+1
    poly = PF(degree=i)
    X_ = poly.fit_transform(np.array(time).reshape(-1, 1))
    # 训练数据的拟合
    LinearR_ = LinearRegression().fit(X_, al)
    X_curve = np.linspace(0,300,500).reshape(-1,1)
    X_curve1 = PF(degree=i).fit_transform(X_curve)
    plt.scatter(time,al,color='k')
    y_curve = LinearR_.predict(X_curve1)
    plt.plot(X_curve,y_curve,label='degree{a}'.format(a = i))
    coef.append(LinearR_.coef_)
    score.append(LinearR_.score(X_,al))
    plt.title("碳数为4-12脂肪醇随时间变化拟合图像")
    plt.xlabel("时间")
    plt.ylabel("碳数为4-12脂肪醇")
    plt.legend()
    plt.show()
a = range(1,10)
print([*zip(a,coef,score)])


fig = plt.figure(figsize=(15, 8), dpi=100)
al = data['乙醛选择性']
score =[]
coef = []
for i in range(0,4):
    i  =i+1
    poly = PF(degree=i)
    X_ = poly.fit_transform(np.array(time).reshape(-1, 1))
    # 训练数据的拟合
    LinearR_ = LinearRegression().fit(X_, al)
    X_curve = np.linspace(0,300,500).reshape(-1,1)
    X_curve1 = PF(degree=i).fit_transform(X_curve)
    plt.scatter(time,al,color='k')
    y_curve = LinearR_.predict(X_curve1)
    plt.plot(X_curve,y_curve,label='degree{a}'.format(a = i))
    coef.append(LinearR_.coef_)
    score.append(LinearR_.score(X_,al))
    plt.title("乙醛选择性随时间变化拟合图像")
    plt.xlabel("时间")
    plt.ylabel("乙醛选择性")
    plt.legend()
    plt.show()
a = range(1,10)
print([*zip(a,coef,score)])



from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression
fig = plt.figure(figsize=(15, 8), dpi=100)
al = data['乙烯选择性']
i = 3
poly = PF(degree=i)
X_ = poly.fit_transform(np.array(time).reshape(-1, 1))
# 训练数据的拟合
LinearR_ = LinearRegression().fit(X_, al)
X_curve = np.linspace(0,450,5000).reshape(-1,1)
X_curve1 = PF(degree=i).fit_transform(X_curve)
plt.scatter(time,al,color='k')
y_curve = LinearR_.predict(X_curve1)
plt.plot(X_curve,y_curve,label='degree{a}'.format(a = i))
print(LinearR_.coef_)
print(LinearR_.score(X_,al))
plt.title("乙烯选择性随时间变化拟合图像")
plt.xlabel("时间")
plt.ylabel("乙烯选择性")
plt.legend()
plt.show()



from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression
fig = plt.figure(figsize=(15, 8), dpi=100)
al = data['乙醛选择性']
i = 3
poly = PF(degree=i)
X_ = poly.fit_transform(np.array(time).reshape(-1, 1))
# 训练数据的拟合
LinearR_ = LinearRegression().fit(X_, al)
X_curve = np.linspace(0,450,5000).reshape(-1,1)
X_curve1 = PF(degree=i).fit_transform(X_curve)
plt.scatter(time,al,color='k')
y_curve = LinearR_.predict(X_curve1)
plt.plot(X_curve,y_curve,label='degree{a}'.format(a = i))
print(LinearR_.coef_)
print(LinearR_.score(X_,al))
plt.title("乙醛选择性随时间变化拟合图像")
plt.xlabel("时间")
plt.ylabel("乙醛选择性")
plt.legend()
plt.show()






