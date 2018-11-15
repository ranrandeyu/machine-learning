from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#n_components主成分的个数，即降维后数据的维度
data=load_iris()#以字典的形式加载鸢尾花的数据集
#print(data)
y=data.target
#print(y)
x=data.data
pca=PCA(n_components=2)#加载PCA算法，设置降维后主成分数目为2
#print(pca)svd_solver='auto'设置特征值分解的方法
reduced_x=pca.fit_transform(x)#对原始数据进行降维，保存在reduced_x中
#print(reduced_x)

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
print('1')
for i in range(len(reduced_x)):
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='*')
plt.show()