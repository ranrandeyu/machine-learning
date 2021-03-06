import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
dataset_x=[]
dataset_y=[]
fr=open('price.txt','r')
lines=fr.readlines()
for line in lines:
    items=line.strip().split(',')
    dataset_x.append(int(items[0]))
    dataset_y.append(int(items[1]))
length=len(dataset_x)
#将datasets_x转化为数组，并变为二维，已符合线性回归拟合函数输入参数要求
dataset_x=np.array(dataset_x).reshape([length,1])
dataset_y=np.array(dataset_y)
minx=min(dataset_x)
maxx=min(dataset_x)
x=np.array(minx,maxx).reshape([-1,1])
#建立datasets_x的二次多项式特征x_poly
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(dataset_x)
lin_reg_2=linear_model.LinearRegression()
lin_reg_2.fit(x_poly,dataset_y)

plt.scatter(dataset_x,dataset_y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x),color='blue'))
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()