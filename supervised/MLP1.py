import numpy as np
#使用listdir模块，用于访问本地文件
from os import listdir
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors

#将加载的32*32的图片矩阵展开成一列向量
def img2vector(filename):
    retmat=np.zeros([1024],int)
    fr=open(filename)
    lines=fr.readlines()
    for i in range(32):
        for j in range(32):
            retmat[i*32+j]=lines[i][j]
    return retmat

#定义加载训练数据的函数readdataset，并将样本标签转换为one-hot
def readdataset(path):
    filelist=listdir(path)
    numfiles=len(filelist)
    #用于存放所有的数字文件
    dataset=np.zeros([numfiles,1024],int)
    #用于存放对应的标签one-hot
    hwlables=np.zeros([numfiles,10])
    for i in range(numfiles):
        filepath=filelist[i]
        #获取文件名称/路径
        digit=int(filepath.split('_')[0])
        #将对应的one-hot标签置1
        hwlables[i][digit]=1.0
        #读取文件内容
        dataset[i]=img2vector(path+'/'+filepath)
    return dataset,hwlables
#hidden_layer_sizes存放一个远足，表示第i层隐藏层中神经元的个数，logistic激活函数和adam优化方法，并令初始学习率为0.00001，迭代2000
clf=MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver='adam',learning_rate_init=0.0001,max_iter=2000)

dataset,hwlables=readdataset('test')
res=clf.predict(dataset)
error_num=0
num=len(dataset)
for i in range(num):
    if np.sum(res[i]==hwlables[i])<10:
        error_num+=1
print("total num",num,'wrong num',error_num,'wrongrate',error_num/float(num))

train_dataset,train_hwlables=readdataset('trainingdights')
knn=neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=3)
knn.fit(train_dataset,train_hwlables)

res2=knn.predict(dataset)
error_num2=np.sum(res!=hwlables)
num=len(dataset)
print('Total num:',num,"Wrong num:",error_num2)