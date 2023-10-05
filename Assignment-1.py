# 导入相关包
import pandas as pd
import numpy as np
import time
# 导入模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# 数据分割包
from sklearn.model_selection import train_test_split
# 评价包
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
# 规范化数据包
from sklearn import preprocessing
# KNN调优包
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# 记录开始时间
start = time.time()

# 读取数据，将缺失值转换为空值的可识别形式
data = pd.read_csv('/Users/zhangx/Desktop/census+income/total.csv')  #total.csv为数据的合集
data = data.replace(' ?', np.NaN)
print(data.head(20))  #打印前20行，观察数据

# 统计并展示每列缺失值的数量
null_all = data.isnull().sum()
print(null_all)

# 删除包含缺失值的行
data = data.dropna(axis=0, how='any')

# 数据探索
print(data.shape)  #数据的行数和列数
print(data.info())  #数据没有缺失值
print(data.describe())  #探索数据

# 数据规范化
x = data.iloc[:, :13]
y = data.iloc[:, 14]
ss = preprocessing.OneHotEncoder()
ss.fit(x)
ss_x = ss.transform(x).toarray()
train_x, test_x, train_y, test_y = train_test_split(ss_x, y, test_size=0.25, random_state=1)

# KNN模型
knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(train_x, train_y)
predict_y = knn.predict(test_x)
# 评分
print("knn_mean_squared_error: ", mean_squared_error(test_y, predict_y))
print("knn_accuracy_score：", accuracy_score(test_y, predict_y))
print("knn_precision_score: ", precision_score(test_y, predict_y))
print("knn_roc_auc_score: ", roc_auc_score(test_y, predict_y))
# 调优
estimator = KNeighborsClassifier()
param = {"n_neighbors": [i for i in range(1, 11)]}
gc = GridSearchCV(estimator, param_grid=param, cv=10)
gc.fit(train_x, train_y)
# 预测准确率
print("Accuracy on the test set:", gc.score(test_x, test_y))
# 输出最优超参数k在交叉验证中的准确率
print("Best results in cross-validation:", gc.best_score_)
# 输出模型参数，包括选出的最优K值
print("Choose the best model is:", gc.best_estimator_)

# 决策树模型
dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_x, train_y)
predict_y = dtc.predict(test_x)
# 评分
print("dtc_mean_squared_error: ", mean_squared_error(test_y, predict_y))
print("dtc_accuracy_score：", accuracy_score(test_y, predict_y))
print("dtc_precision_score: ", precision_score(test_y, predict_y))
print("dtc_roc_auc_score: ", roc_auc_score(test_y, predict_y))

end = time.time()
print("Running_Time: {}".format(end-start))  #打印运行时间