import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def loadData(filePath):
    f=open(filePath,'rb')
    data=[]
    img=image.open(f)#一列表的形式返回图片像素值
    m,n=img.size
    for i in range(m):
        for j in range(n):
            x,y,z=img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    print(data)
    return np.mat(data),m,n
imgData,row,col=loadData('1.jpg')

#对像素点进行聚类并输出
#聚类获得每个像素所属的类别
label=KMeans(n_clusters=6).fit_predict(imgData)
label=label.reshape([row,col])
#创建一张新的灰度图保存聚类后的结果
pic_new=image.new("L",(row,col))
#根据所属类别向图像添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
pic_new.save('2.jpg','JPEG')

