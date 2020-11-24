# author: xinyuan time: 23/11/2020.
from sklearn import datasets
from sklearn.datasets import load_iris
from pandas import DataFrame
import pandas as pd

data = load_iris();
iris = load_iris()
print("鸢尾花数据集：\n", iris)
print("查看数据集描述：\n", iris["DESCR"])
print("查看特征值的名字：\n", iris.feature_names)
print("查看数据集的shape：\n", iris.data.shape)
x_data = datasets.load_iris().data # 返回iris的所有特征
y_data = datasets.load_iris().target # 返回iris的所有标签
print("x_data from dataset: \n", x_data)
print("y_data from dataset: \n", y_data)

x_data = DataFrame(x_data, columns=["花萼长度","花萼宽度","花瓣长度","花瓣宽度"])
print("x_data: \n", x_data)
pd.set_option("display.unicode.east_asian_width", True) # 设置列名对齐
print("x_data add index: \n", x_data)

x_data["类别"] = y_data #新加一列，列名标签为“类别”， 数据为y_data
print("x_data add a column: \n", y_data)
pd.set_option("display.unicode.east_asian_width", True) # 设置列名对齐
print(x_data)



