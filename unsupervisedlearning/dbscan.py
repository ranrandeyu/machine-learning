'''
数据介绍：
现有大学校园网的日志数据，290条大学生的校园网使用情况数据，数据包括用户ID，设备的MAC地址，IP地址，开始上网时间，停止上网时间，上网时长，校园网套餐等。 利用已有数据，分析学生上网的模式。

实验目的：
通过DBSCAN聚类，分析学生上网时间和上网时长的模式。'''
import numpy as np
import  sklearn.cluster as skc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
#esp两个样本被看成邻居节点的最大距离,min_sample簇的样本数，metric距离计算方式Euclidean（欧式距离）
mac2id=dict()#字典
f=open('time.txt','r',encoding='utf-8')
#print(f)
onlinetimes=[]
for line in f:
    mac=line.split(',')[0]
    starttime=line.split(',')[1].split(' ')[1].split(':')[0]
    onlinetime=line.split(',')[2]
    if mac not in mac2id:
        mac2id[mac]=len(onlinetime)
        onlinetimes.append((starttime,onlinetime))#获得开始时间点以及总时间数[('20', '1228\n'), ('21', '1234\n'), ('11', '213\n'), ('21', '313\n'), ('20', '1228')]
    else:
        onlinetimes[mac2id[mac]]=((starttime,onlinetime))
print(onlinetimes)
real_x=np.array(onlinetimes).reshape((-1,2))#一个参数为-1时，那么reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值。
print(real_x)
stime=real_x[:,0:1]#只得到上网时间【每行，列的第0个】
print(stime)

#调用DBSCAN方法进行训练，labels为每个数据的簇标签
db=DBSCAN(eps=0.01,min_samples=2).fit(stime)
lables=db.labels_#返回的数据的簇标签，噪声数据标签为-1
#print("Lable:\n",lables)

#print(lables[:])
#print(lables)
#计算标签为-1的数据（即噪声数据）的比例
radio=len(lables[lables==-1])/len(lables)#lables跟lables[:]一样的
print('Noise radio:',format(radio,'.2%'))#转换为百分比
#print('hello {} {}'.format('lacey','!')) hello lacey !

#计算簇的个数
n_clusters_=len(set(lables))-(1 if -1 in lables else 0)#判断标签里面是否存在-1，如果存在则为1，不存在则为-1
#print(n_clusters_)
#si 接近1说明是聚类合理，-1分类到另外的簇，0说明在两个簇的边界
print('Estimated number of cluster:%d'%n_clusters_)
print('Silhouette Coefficient :%0.3f'%metrics.silhouette_score(stime,lables))#聚类效果评价指标

#各簇标号以及各簇内的数据
for i in range(n_clusters_):
    print('number of data in Cluster %s is :%s',i,len(stime[lables==i]))
plt.hist(stime,24)
plt.show()