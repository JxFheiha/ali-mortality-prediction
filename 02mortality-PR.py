from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score, auc, roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import os



if __name__ == '__main__':
    file_path1 = "E:/ML/ALI-stroke/pythonML/202508revise/ALImortality-train-transformed-data.xlsx"
    print(file_path1)
    df = pd.read_excel(file_path1)

    features1 = set(df.columns)
    features1.remove("Mortality")
    features1 = list(features1)
    X_train = np.array(df[features1])
    print(X_train.shape)
    y_train = df["Mortality"]
    #
    file_path2 = "E:/ML/ALI-stroke/pythonML/202508revise/ALImortality-test-transformed-data.xlsx"
    print(file_path2)
    df = pd.read_excel(file_path2)
    features2 = set(df.columns)
    features2.remove("Mortality")
    features2 = list(features2)
    X_test = np.array(df[features2])
    print(X_test.shape)
    y_test = df["Mortality"]


    #构建模型
    lr = LogisticRegression(dual=False, tol=0.1, C=1,random_state=None, solver='liblinear', max_iter=25, n_jobs=1)
    lr.fit(X_train, y_train)
    lr_y_proba = lr.predict_proba(X_test)#预测属于某标签的概率
    lr_y_pre = lr.predict(X_test)#预测标签
   
    
    rf = RandomForestClassifier(bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300)
    rf.fit(X_train, y_train)
    rf_y_pre = rf.predict(X_test)
    rf_y_proba = rf.predict_proba(X_test)

    tr = DecisionTreeClassifier()
    tr.fit(X_train, y_train)
    tr_y_pre = tr.predict(X_test)
    tr_y_proba = tr.predict_proba(X_test)

    Xgbc = XGBClassifier(learning_rate =0.01, n_estimators=100,min_child_weight=1, 
                            subsample=0.6, colsample_bytree=0.02, seed=100)
    Xgbc.fit(X_train, y_train)
    Xgbc_y_pre = Xgbc.predict(X_test)
    Xgbc_y_proba = Xgbc.predict_proba(X_test)


    svm = SVC(C=1,random_state=None,probability=True)
    svm.fit(X_train, y_train)
    svm_y_pre = svm.predict(X_test)
    svm_y_proba = svm. predict_proba(X_test)

    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_y_pre = knn.predict(X_test)
    knn_y_proba = knn.predict_proba(X_test)



# 存储每个模型的AUC-PR值
auc_pr_values = {}


# 创建一个图形
plt.figure(figsize=(10, 7))

# 绘制每个模型的Precision-Recall曲线
for model_name, y_proba in [("LR", lr_y_proba),
                            ("DT", tr_y_proba),
                            ("XGB", Xgbc_y_proba),
                            ("SVM", svm_y_proba),
                            ("KNN", knn_y_proba),
                            ("RF", rf_y_proba),]:

# 计算Precision和Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
    # 计算AUC-PR
    auc_pr = auc(recall, precision)
    auc_pr_values[model_name] = auc_pr
    # 绘制曲线
    if model_name == "LR":
        color = '#0000cc'
    elif model_name == "DT":
        color = '#ff0000'
    elif model_name == "XGB":
        color = '#006633'
    elif model_name == "SVM":
        color = '#000000'
    elif model_name == "KNN":
        color = '#663300'
    elif model_name == "RF":
        color = '#ff0099'
    plt.plot(recall, precision, color=color, linewidth=1.3, label=f'{model_name} (AP: {auc_pr:.3f})')

# 设置图形细节
plt.title('Mortality-Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="best")
plt.grid(False)


# 打印AUC-PR值
for model, auc_pr in auc_pr_values.items():
    print(f'{model} AUC-PR: {auc_pr:.6f}')

# 显示图形
plt.show()

    
