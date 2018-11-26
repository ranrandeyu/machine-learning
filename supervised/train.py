#knn
from sklearn.neighbors import KNeighborsClassifier
x=[[0],[1],[2],[3]]
y=[0,0,1,1]
#最近3个邻居作为分类依据
neight=KNeighborsClassifier(n_neighbors=3)
#fit函数，进行学习训练
neight.fit(x,y)
#不知道标签的新样本predict（）函数，对于未知分类样本【1.1】分类，样本取0，1，2这3个邻居作为依据
print(neight.predict([[1.1]]))

#decisiontree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score#计算交叉验证值
#使用默认参数，创建一颗基于基尼系数的决策树，并将该决策树分类器赋值给变量clf
clf=DecisionTreeClassifier()
#鸢尾花数据赋值给iris
iris=load_iris()
#交叉验证的得分
print(cross_val_score(clf,iris.data,iris.target,cv=10))
#利用fit（）函数训练模型并使用predict（）函数预测
clf.fit(X,y)
clf.predict(x)

#naive_bayes
import numpy as np
from sklearn.naive_bayes import  GaussianNB
X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1]])
Y=np.array([1,1,1,1,2])
clf=GaussianNB(priors=None)
print(clf)
clf.fit(X,Y)
print(clf.predict([[0.2,1]]))
