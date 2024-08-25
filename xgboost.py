import shap
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 用于画图
import shap
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score


df = pd.read_csv('D:\mlx_jqxx\input.csv') #导入数据
print(df.head())  # 输出表的前五行
print('数据维度：', df.shape)  # 输出表格的规模

x = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=81)

xgbr = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.03, gamma=0, min_child_weight=1, subsample=0.999,random_state=81)#树的数目300一下为佳
# max_depth=4, learning_rate=0.3, gamma=0, min_child_weight=1, subsample=1
xgbr.fit(x_train, y_train)
y_train_pred = xgbr.predict(x_train)
y_test_pred = xgbr.predict(x_test)
print(y_test_pred)


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
explainer = shap.TreeExplainer(xgbr, x_train)
shap_values = explainer(x_train)

# # 特征重要性图
shap.summary_plot(shap_values, x_train, plot_type="bar", max_display=30, show=False)  # 特征重要性图,更改时候要show=False
plt.xticks(fontproperties='Times New Roman', size=10)  # 定义字体及大小
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xlabel('Mean(|SHAP value|)', fontproperties='Times New Roman', fontsize=15)  # 定义xlabel
plt.gcf().subplots_adjust(left=0.15, top=0.97, bottom=0.079) #调整图片位置，使其全显示
# #保存为pdf
fig = plt.gcf()
fig.set_size_inches(5, 7)
plt.savefig("D:\mlx_jqxx\summary_shap_xgboost.pdf", bbox_inches="tight") #保存为pdf
plt.show()  # 出图

# #
shap.summary_plot(shap_values, x_train,max_display=30, show=False)  # 特征密度散点图
plt.xticks( fontproperties='Times New Roman', size=10) #定义字体及大小
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xlabel('SHAP value(impact on model output)', fontproperties='Times New Roman', fontsize=15) #定义xlabel
plt.gcf().subplots_adjust(left=0.17, top=0.97, bottom=0.09) #调整图片位置，使其全显示
# #保存为pdf
fig = plt.gcf()
fig.set_size_inches(6.5, 7)
plt.savefig("D:\mlx_jqxx\scatter_shap_xgboost.pdf", bbox_inches="tight") #保存为pdf
plt.show() #出图


# #特征依赖图
shap.dependence_plot("Tur", shap_values.values, x_train, interaction_index='COD', dot_size=110,show=False)
plt.xticks( fontproperties='Times New Roman', size=10) #定义字体及大小
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xlabel('Tur', fontproperties='Times New Roman', fontsize=15) #定义xlabel
plt.ylabel('SHAP value for Tur', fontproperties='Times New Roman', fontsize=15) #定义xlabel
plt.gcf().subplots_adjust(left=0.2, top=0.97, bottom=0.15) #调整图片位置，使其全显示
# #保存为pdf
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.savefig("D:\mlx_jqxx\dependence_shap_XG_Tur+COD.pdf", bbox_inches="tight") #保存为pdf
plt.show() #出图