from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归模型
from sklearn.datasets import make_blobs  # 导入聚类生成器
from sklearn import metrics  # 可以输出准确率、敏感度、Kappa值等等
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 用于画图
import openpyxl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import shap
from shap.plots import _waterfall
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv('D:\mlx_jqxx\input.csv') # 导入数据
print(df.head())  # 输出表的前五行
print('数据维度：', df.shape)  # 输出表格的规模
#
x = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
print(x)
print(y)
# '''
# 自变量x为表格除最后一列的数据，因变量为最后一列的数据
# '''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=81)
# 划分测试集和训练集，并且规定随机参数

forest = RandomForestRegressor(n_estimators=600, max_depth=7, min_samples_split=2, min_samples_leaf=1,criterion='squared_error', random_state=81, n_jobs=1)

forest.fit(x_train, y_train)
y_train_pred = forest.predict(x_train)
y_test_pred = forest.predict(x_test)
print(y_test_pred)

result = forest.score(x_test, y_test) #result在randomforest regressor中为r2
print(result)

print('MSE train:%.3f,test:%.3f' % (
    metrics.mean_squared_error(y_train, y_train_pred), metrics.mean_squared_error(y_test, y_test_pred)))
print('R^2 train:%.3f,test:%.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
print(
    'MAE train:%.3f,test:%.3f' % (mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)))
print('RMSE train:%.3f,test:%.3f' % (
    sqrt(metrics.mean_squared_error(y_train, y_train_pred)), sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print('MAPE train:%.3f,test:%.3f' % (
    metrics.mean_absolute_percentage_error(y_train, y_train_pred), mean_absolute_percentage_error(y_test, y_test_pred)))

# shape values
explainer = shap.TreeExplainer(forest, x_train)
shap_values = explainer(x_train)

# #

# 特征重要性图
shap.summary_plot(shap_values, x_train, plot_type="bar", max_display=30, show=False) #特征重要性图,更改时候要show=False
plt.xticks( fontproperties='Times New Roman', size=10) #定义字体及大小
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xlabel('Mean(|SHAP value|)', fontproperties='Times New Roman', fontsize=15) #定义xlabel
plt.gcf().subplots_adjust(left=0.13, top=0.97, bottom=0.09) #调整图片位置，使其全显示
# #保存为pdf
fig = plt.gcf()
fig.set_size_inches(5, 7)
plt.savefig("D:\mlx_jqxx\summary_shap_RF.pdf", bbox_inches="tight") #保存为pdf
plt.show() #出图

# # # # # '''

shap.summary_plot(shap_values, x_train, max_display=10, show=False)  # 特征密度散点图
plt.xticks( fontproperties='Times New Roman', size=10) #定义字体及大小
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xlabel('SHAP value (impact on model output)', fontproperties='Times New Roman', fontsize=15) #定义xlabel
plt.gcf().subplots_adjust(left=0.2, top=0.97, bottom=0.09) #调整图片位置，使其全显示
# #保存为pdf
fig = plt.gcf()
fig.set_size_inches(6.5, 7)
plt.savefig("D:\mlx_jqxx\scatter_shap_RF.pdf", bbox_inches="tight") #保存为pdf
plt.show() #出图

# # 特征依赖图
shap.dependence_plot("Turbidity", shap_values.values, x_train, interaction_index='COD', dot_size=110, show=False)
plt.xticks( fontproperties='Times New Roman', size=10) #定义字体及大小
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xlabel('Turbidity', fontproperties='Times New Roman', fontsize=15) #定义xlabel
plt.ylabel('SHAP value for Turbidity', fontproperties='Times New Roman', fontsize=15) #定义xlabel
plt.gcf().subplots_adjust(left=0.2, top=0.97, bottom=0.19) #调整图片位置，使其全显示
# #保存为pdf
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.savefig("D:\mlx_jqxx\dependence_shap_RF_Tur+COD.pdf", bbox_inches="tight") #保存为pdf
plt.show() #出图