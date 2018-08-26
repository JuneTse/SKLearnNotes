#coding: utf-8
import numpy as np
from sklearn.manifold.t_sne import TSNE
import matplotlib.pyplot as plt

c=TSNE(n_components=2,
       perplexity=30,
       early_exaggeration=4.0,
       learning_rate=1000.0,
       n_iter=1000,
       n_iter_without_progress=100,
       min_grad_norm=1e-7)

data=np.random.rand(50,200)
labels=list(range(50))

#降维
res=c.fit_transform(data)
print(res.shape)
#可视化
fig=plt.figure(figsize=(8,8))
#遍历每个点使用scatter画点，使用annotate画点的标签
for (x,y),label in zip(res,labels):
    plt.scatter(x,y,color='red',lw=2,label=labels)
    plt.annotate(s=label,
                 xy=(x,y),
                 xytext=(5,2), #文本的位置，与textcoords配合使用，offset points表示相对点坐标的偏移量，否则是绝对位置
                 textcoords='offset points', #文本的坐标，可能与点的坐标不同
                 ha='right',
                 va='bottom')
