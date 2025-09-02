from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score, auc, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import os
import seaborn as sns


if __name__ == '__main__':
    file_path = "E:/ML/ALI-stroke/pythonML/202508revise/ALI-mortality-preprocess-revise.xlsx"
    print(file_path)
    data = pd.read_excel(file_path)
output_path = 'E:/ML/ALI-stroke/pythonML/202508revise'  

# 定义特征和目标变量
X = data[['Gender', 'Age', 'Ethnicity', 'Education level', 'Marital status', 'PIR', 'BMI', 'CVD', 'DM', 'Hypertension', 'Smoking status', 'HbA1c', 'ALT', 'AST', 'BUN', 'TC', 'HDL', 'Drinking status', 'ALI']]
y = data['Stroke']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=4, max_features="sqrt",
                                oob_score=True, random_state=10, bootstrap=True, criterion="gini")

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 计算性能指标
auc_roc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f'AUC-ROC: {auc_roc}, F1 Score: {f1}, Specificity: {specificity}, Sensitivity: {sensitivity}')

# 获取特征重要性
importances = rf.feature_importances_

# 创建特征重要性DataFrame
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# 输出到Excel文件
wb = Workbook()
ws = wb.active
ws.title = "RF Feature Importances"

for index, row in feature_importances.iterrows():
    ws.append([row['Feature'], row['Importance']])

wb.save('Mortality-feature_importances.xlsx')
