import numpy as np
import pandas as pd
#预处理函数imputer
from sklearn.preprocessing import Imputer
#自动生成训练集和测试集的模块
from sklearn.model_selection import train_test_split
#预测结果评估
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#数据导入函数
def load_dataset(feature_paths,label_paths):
    feature=np.ndarray(shape=(0,41))
    label=np.ndarray(shape=(0,1))
    for file in feature_paths:
        #使用，分割读取数据特征，将问号替代标记为缺失值，文件中不包括表头
        df=pd.read_table(file,delimiter=',',na_values='?',header=None)
        #使用平均值strategy='mean'填补缺失值，数据补齐,axis=0表示行
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
        #fit训练预处理器
        imp.fit(df)
        #transform生成预处理结果
        df=imp.transform(df)
        #将新读入的数据合并到特征集合中
        feature=np.concatenate((feature,df))
    for file in label_paths:
        df=pd.read_table(file,header=None)
        label=np.concatenate((label,df))
    label=np.ravel(label)
    return feature,label

if __name__=='__main__':
    #设置数据路径
    feature_paths=['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
    label_paths=['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
    x_train,y_train=load_dataset((feature_paths[:4],label_paths[:4]))
    x_test,y_test=load_dataset(feature_paths[4:],label_paths[4:])
    #使用全量数据作为数据集，train_test_split函数将训练数据打乱
    x_train,x_,y_train,y_=train_test_split(x_train,y_train,test_size=0.0)

    print("Start training KNN")
    knn=KNeighborsClassifier().fit(x_train,y_train)
    print("Training done")
    answer_knn=knn.predict(x_test)
    print("Prediction done")
    #classification_report函数对分类结果，从精确率precision，召回率recall、f1值f1-score和支持度support
    print(classification_report(y_test,answer_knn))

    print("Start training DT")
    dt=DecisionTreeClassifier().fit(x_train,y_train)
    answer_dt=dt.predict(x_test)
    print(classification_report(y_test, answer_dt))

    gnb=GaussianNB().fit(x_train,y_train)
    answer_gnb=gnb.predict(x_test)
    print(classification_report(y_test, answer_gnb))