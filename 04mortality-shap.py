#Import libraries:
import pandas as pd
import numpy as np
import sklearn
from shap import force_plot
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import lightgbm as lgb
import re
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from sklearn.ensemble import RandomForestClassifier#随机森林  RandomForestClassifier()
import shap
shap.initjs()



df1 = pd.read_csv("E:/ML/ALI-stroke/pythonML/202508revise/ALI-mortality-preprocess-revise-SHAPanalysis.csv", encoding='gbk')
data=df1
# GBM建模数据准备=====================================================
# 第一列为输出结果
data_result = data.iloc[:,0]
# 第2-6为特征值
data_input = data.iloc[:,1:20]
print(data_result)

from sklearn.model_selection import train_test_split
# 准备GBM训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data_input, data_result, test_size=0.3, random_state=42)

# 查看训练集和测试集的特征值形状
print(train_x.shape, test_x.shape)
# 查看训练集各类型选择的抽样数量
train_y.value_counts()

print(train_x)
print(train_y)


#训练参数设置
model = RandomForestClassifier(bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300)
model.fit(train_x,train_y)

##shap解释
explainer = shap.TreeExplainer(model) # #这里的model在准备工作中已经完成建模，模型名称就是model
shap_values = explainer.shap_values(test_x) # 传入特征矩阵X，计算SHAP值


# 展示force plot
shap.initjs()
force_plot_one =shap.force_plot(explainer.expected_value[1], shap_values[0][0, :], test_x.iloc[0, :])
shap.save_html("Mortality-force_plot_one-1.html", force_plot_one)



# summarize the effects of all the features
shap.summary_plot(shap_values[1], test_x,show=False)#max_display=10,
plt.savefig('Mortality-shapsummary.pdf', bbox_inches="tight",dpi=300)
plt.clf()
plt.close()

shap.summary_plot(shap_values, test_x, plot_type="bar", show=False)
plt.savefig('Mortality-shap柱状.pdf', bbox_inches='tight', dpi=300)


