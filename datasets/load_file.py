#coding:utf-8
from sklearn.datasets import load_svmlight_file,load_svmlight_files

"""
1. 加载libsvm格式的数据集
    * 使用datasets中的load_svmlight_file
        * 返回：
            X: scipy.sparse 矩阵
            y: n_samples个标签或标签list
    * 适合处理稀疏数据
2. 保存为libsvm格式的数据
    * 使用datasets中的dump_svmlight_file
"""

X_train,y_train=load_svmlight_file('train.txt')

print(X_train.shape)


