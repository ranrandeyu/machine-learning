'''数据介绍：
现有1999年全国31个省份城镇居民家庭平均每人全年消费性支出的八个主要变量数据，这八个变量分别是：食品、 衣着、 家庭设备用品及服务、 医疗保健、 交通和通讯、 娱乐教育文化服务、 居住以及杂项商品和服务。 利用已有数据，对31个省份进行聚类。

实验目的：
通过聚类，了解1999年各个省份的消费水平在国内的情况。'''
#建立工程，导入sklearn相关包
import numpy as np
from sklearn.cluster import KMeans#默认的是欧式距离

def loadData(filepath):
    #读写打开一个文本文件
    fr=open(filepath,'r+',encoding='utf-8')#r+读写的方式打开一个文本文件
    lines=fr.readlines()#一次读取整个文件
    retdata=[]
    retcityname=[]
    for line in lines:
        items=line.strip().split(',')#strip()移除字符串头尾指定的字符(默认为空格或换行符)或字符序列。
        retcityname.append(items[0])
        retdata.append([float(items[i]) for i in range(1,len(items))])
        #print(retdata)
    return retdata,retcityname
#加载数据，创建K-means算法实例，进行训练，获得标签
if __name__ == '__main__':
    data,cityname=loadData('kmeans-city.txt')
    print(data)
    km=KMeans(n_clusters=3)#簇中心点数为3个
    lable=km.fit_predict(data)#lable聚类后各数据所属的标签
    print(lable)
    expense=np.sum(km.cluster_centers_,axis=1)#聚类中心点的数值加和，数值上等于平均值也就是平均消费水平
    print(expense)
    citycluster=[[],[],[]]
    for i in range(len(cityname)):
        citycluster[lable[i]].append(cityname[i])#将相同簇的cityname放在一起
    for i in range(len(citycluster)):
        print("Expenses:%.2f"%expense[i])
        print(citycluster[i])






