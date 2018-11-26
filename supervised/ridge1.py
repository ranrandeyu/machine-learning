import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sklearn.cross_validation
from sklearn.preprocessing import PolynomialFeatures
data=np.genfromtxt('data.txt')
plt.plot(data[:,4])
#x用于保存0-3维数据，即属性
x=data[:,:4]
#y用于保存4维数据，即车流量
y=data[:,4]
#用于创建最高次数6次方的多项式特征
poly=PolynomialFeatures(6)
X=poly.fit_transform(x)
test_set_X,test_set_Y,train_set_x,train_set_y=sklearn.cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
clf=Ridge(alpha=1.0,fit_intercept=True)
#调用fit函数使用训练集训练回归器
clf.fit(train_set_x,train_set_y)
#clf.score计算回归曲线的拟合优度
clf.score(test_set_X,test_set_Y)

start=200
end=300
y_pre=clf.predict(X)
time=np.arange(start,end)
plt.plot(time,y[start:end],'b',label='real')
plt.plot(time,y_pre[start,end],'r',label='predict')
plt.legend(loc='upper left')
plt.show()