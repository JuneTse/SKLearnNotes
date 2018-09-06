#coding:utf-8
from sklearn.datasets import load_boston,load_digits,load_iris
from sklearn.datasets import dump_svmlight_file

'''
1. 加载toy数据集
'''

X_iris,y_iris=load_iris(return_X_y=True)
#保存为lightsvm格式
dump_svmlight_file(X_iris,y_iris,open("train.txt",'wb'))

X_digits,y_digits=load_digits(n_class=10,return_X_y=True)
print("X:",X_digits.shape)
print("y:",y_digits.shape)