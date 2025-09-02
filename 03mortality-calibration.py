from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d
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
    #
    #
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_y_pre = knn.predict(X_test)
    knn_y_proba = knn.predict_proba(X_test)




# 计算训练集校准曲线
lr_prob_true, lr_prob_pred = calibration_curve(y_test, lr_y_proba[:, 1], n_bins=10)
rf_prob_true, rf_prob_pred = calibration_curve(y_test, rf_y_proba[:, 1], n_bins=10)
tr_prob_true, tr_prob_pred = calibration_curve(y_test, tr_y_proba[:, 1], n_bins=10)
Xgbc_prob_true, Xgbc_prob_pred = calibration_curve(y_test, Xgbc_y_proba[:, 1], n_bins=10)
svm_prob_true, svm_prob_pred = calibration_curve(y_test, svm_y_proba[:, 1], n_bins=10)
knn_prob_true, knn_prob_pred = calibration_curve(y_test, knn_y_proba[:, 1], n_bins=10)


# 计算Brier Score
lr_brier_score = brier_score_loss(y_test, lr_y_proba[:, 1], pos_label=1)
rf_brier_score = brier_score_loss(y_test, rf_y_proba[:, 1], pos_label=1)
tr_brier_score = brier_score_loss(y_test, tr_y_proba[:, 1], pos_label=1)
Xgbc_brier_score = brier_score_loss(y_test, Xgbc_y_proba[:, 1], pos_label=1)
svm_brier_score = brier_score_loss(y_test, svm_y_proba[:, 1], pos_label=1)
knn_brier_score = brier_score_loss(y_test, knn_y_proba[:, 1], pos_label=1)


# 绘制校准曲线
plt.figure(figsize=(10, 7))
plt.plot(lr_prob_pred, lr_prob_true,lw=1.3, label='LR (Brier: {:.3f})'.format(lr_brier_score) , color='#0000cc')
plt.plot(tr_prob_pred, tr_prob_true,lw=1.3, label='DT (Brier Score: {:.3f})'.format(tr_brier_score), color='#ff0000')
plt.plot(Xgbc_prob_pred, Xgbc_prob_true, lw=1.3,label='XGB (Brier Score: {:.3f})'.format(Xgbc_brier_score), color='#006633')
plt.plot(svm_prob_pred, svm_prob_true, lw=1.3,label='SVM (Brier Score: {:.3f})'.format(svm_brier_score), color='#000000')
plt.plot(knn_prob_pred, knn_prob_true, lw=1.3,label='KNN (Brier Score: {:.3f})'.format(knn_brier_score), color='#663300')
plt.plot(rf_prob_pred, rf_prob_true, lw=1.3, label='RF (Brier Score: {:.3f})'.format(rf_brier_score), color='#ff0099')
plt.plot([0, 1], [0, 1], linestyle='--', color='#999999',label='Pefecly calibrated')  # 理想校准曲线
plt.xlabel('Predicted Probability',fontsize=13)
plt.ylabel('True Probability',fontsize=13)
plt.title('Mortality-Calibration Curve',fontsize=13)
plt.legend(loc="lower right")
plt.show()

# 输出结果
print("lr_Brier Score:", lr_brier_score)
print("rf_Brier Score:", rf_brier_score)
print("tr_Brier Score:", rf_brier_score)
print("Xgbc_Brier Score:", Xgbc_brier_score)
print("svm_Brier Score:", svm_brier_score)
print("knn_Brier Score:", knn_brier_score)


print("Test Calibration Curve:")
print("lr_Predicted Probabilities:", lr_prob_pred)
print("lr_True Probabilities:", lr_prob_true)
print("rf_Predicted Probabilities:", rf_prob_pred)
print("rf_True Probabilities:", rf_prob_true)
print("Xgbc_Predicted Probabilities:", Xgbc_prob_pred)
print("Xgbc_True Probabilities:", Xgbc_prob_true)
print("svm_Predicted Probabilities:", svm_prob_pred)
print("svm_True Probabilities:", svm_prob_true)
print("knn_Predicted Probabilities:", knn_prob_pred)
print("knn_True Probabilities:", knn_prob_true)
