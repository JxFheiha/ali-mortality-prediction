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
from sklearn.metrics import confusion_matrix
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

    svm = SVC(C=0.1,random_state=None,probability=True)
    svm.fit(X_train, y_train)
    svm_y_pre = svm.predict(X_test)
    svm_y_proba = svm. predict_proba(X_test)
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_y_pre = knn.predict(X_test)
    knn_y_proba = knn.predict_proba(X_test)
    # #模型评分
    lr_score = lr.score(X_test, y_test)
    lr_accuracy_score = accuracy_score(y_test,lr_y_pre)
    lr_preci_score = precision_score(y_test,lr_y_pre)
    lr_recall_score = recall_score(y_test, lr_y_pre)
    lr_f1_score = f1_score(y_test, lr_y_pre)
    lr_auc = roc_auc_score(y_test, lr_y_proba[:, 1])
    print(lr_accuracy_score,lr_preci_score, lr_recall_score, lr_f1_score, lr_auc)
    print('lr_accuracy_score: %f,lr_preci_score: %f,lr_recall_score: %f,lr_f1_score: %f,lr_auc: %f'
          % (lr_accuracy_score, lr_preci_score, lr_recall_score, lr_f1_score, lr_auc))

    rf_score = rf.score(X_test, y_test)
    rf_accuracy_score = accuracy_score(y_test, rf_y_pre)
    rf_preci_score = precision_score(y_test, rf_y_pre)
    rf_recall_score = recall_score(y_test, rf_y_pre)
    rf_f1_score = f1_score(y_test, rf_y_pre)
    rf_auc = roc_auc_score(y_test, rf_y_proba[:, 1])
    print('rf_accuracy_score: %f,rf_preci_score: %f,rf_recall_score: %f,rf_f1_score: %f,rf_auc: %f'
          % (rf_accuracy_score, rf_preci_score, rf_recall_score, rf_f1_score, rf_auc))

    tr_score = tr.score(X_test, y_test)
    tr_accuracy_score = accuracy_score(y_test,tr_y_pre)
    tr_preci_score = precision_score(y_test,tr_y_pre)
    tr_recall_score = recall_score(y_test,tr_y_pre)
    tr_f1_score = f1_score(y_test,tr_y_pre)
    tr_auc = roc_auc_score(y_test,tr_y_proba[:,1])
    print('tr_accuracy_score: %f,tr_preci_score: %f,tr_recall_score: %f,tr_f1_score: %f,tr_auc: %f'
        % (tr_accuracy_score,tr_preci_score,tr_recall_score,tr_f1_score,tr_auc))
    # #
    Xgbc_score = Xgbc.score(X_test, y_test)
    Xgbc_accuracy_score = accuracy_score(y_test,Xgbc_y_pre)
    Xgbc_preci_score = precision_score(y_test,Xgbc_y_pre)
    Xgbc_recall_score = recall_score(y_test,Xgbc_y_pre)
    Xgbc_f1_score = f1_score(y_test,Xgbc_y_pre)
    Xgbc_auc = roc_auc_score(y_test,Xgbc_y_proba[:,1])
    print('Xgbc_accuracy_score: %f,Xgbc_preci_score: %f,Xgbc_recall_score: %f,Xgbc_f1_score: %f,Xgbc_auc: %f'
        % (Xgbc_accuracy_score,Xgbc_preci_score,Xgbc_recall_score,Xgbc_f1_score,Xgbc_auc))

    svm_score = svm.score(X_test, y_test)
    svm_accuracy_score = accuracy_score(y_test,svm_y_pre)
    svm_preci_score = precision_score(y_test,svm_y_pre)
    svm_recall_score = recall_score(y_test,svm_y_pre)
    svm_f1_score = f1_score(y_test,svm_y_pre)
    svm_auc = roc_auc_score(y_test,svm_y_proba[:,1])
    print('svm_accuracy_score: %f,svm_preci_score: %f,svm_recall_score: %f,svm_f1_score: %f,svm_auc: %f'
        % (svm_accuracy_score,svm_preci_score,svm_recall_score,svm_f1_score,svm_auc))

    knn_score = knn.score(X_test, y_test)
    knn_accuracy_score = accuracy_score(y_test, knn_y_pre)
    knn_preci_score = precision_score(y_test, knn_y_pre)
    knn_recall_score = recall_score(y_test, knn_y_pre)
    knn_f1_score = f1_score(y_test, knn_y_pre)
    knn_auc = roc_auc_score(y_test, knn_y_proba[:, 1])
    print('knn_accuracy_score: %f,knn_preci_score: %f,knn_recall_score: %f,knn_f1_score: %f,knn_auc: %f'
          % (knn_accuracy_score, knn_preci_score, knn_recall_score, knn_f1_score, knn_auc))

#5折交叉验证
    scores_lr = cross_val_score(lr, X_test, y_test, cv=5)
    print("lr", scores_lr.mean())
    scores_rf = cross_val_score(rf, X_test, y_test, cv=5)
    print("rf", scores_rf.mean())
    scores_tr = cross_val_score(tr, X_test, y_test, cv=5)
    print("tr", scores_tr.mean())
    scores_Xgbc = cross_val_score(Xgbc, X_test, y_test, cv=5)
    print("Xgbc", scores_Xgbc.mean())
    scores_svm = cross_val_score(svm, X_test, y_test, cv=5)
    print("svm", scores_svm.mean())
    scores_knn = cross_val_score(knn, X_test, y_test, cv=5)
    print("knn", scores_knn.mean())

    
#混合矩阵
    lr_confusion_matrix = confusion_matrix (y_test, lr_y_pre)
    print(lr, lr_confusion_matrix)
    rf_confusion_matrix = confusion_matrix(y_test, rf_y_pre)
    print(rf, rf_confusion_matrix)
    tr_confusion_matrix = confusion_matrix(y_test, tr_y_pre)
    print(tr, tr_confusion_matrix)
    Xgbc_confusion_matrix = confusion_matrix(y_test, Xgbc_y_pre)
    print(Xgbc, Xgbc_confusion_matrix)
    svm_confusion_matrix = confusion_matrix(y_test, svm_y_pre)
    print(svm, svm_confusion_matrix)
    knn_confusion_matrix = confusion_matrix(y_test, knn_y_pre)
    print(knn, knn_confusion_matrix)

#计算secificity
lr_specificity = lr_confusion_matrix [0,0]/(lr_confusion_matrix[0,0]+lr_confusion_matrix[0,1])
rf_specificity = rf_confusion_matrix [0,0]/(rf_confusion_matrix[0,0]+rf_confusion_matrix[0,1])
tr_specificity = tr_confusion_matrix [0,0]/(tr_confusion_matrix[0,0]+tr_confusion_matrix[0,1])
Xgbc_specificity = Xgbc_confusion_matrix [0,0]/(Xgbc_confusion_matrix[0,0]+Xgbc_confusion_matrix[0,1])
svm_specificity = svm_confusion_matrix [0,0]/(svm_confusion_matrix[0,0]+svm_confusion_matrix[0,1])
knn_specificity = knn_confusion_matrix [0,0]/(knn_confusion_matrix[0,0]+knn_confusion_matrix[0,1])

print('lr_specificity: %f'
        % (lr_specificity))
print('rf_specificity: %f'
        % (rf_specificity))
print('tr_specificity: %f'
        % (tr_specificity))
print('Xgbc_specificity: %f'
        % (Xgbc_specificity))
print('svm_specificity: %f'
        % (svm_specificity))
print('knn_specificity: %f'
        % (knn_specificity))

    #画ROC曲线
lr_fpr, lr_tpr, lr_threasholds = roc_curve(y_test, lr_y_proba[:, 1])
rf_fpr, rf_tpr, rf_threasholds = roc_curve(y_test, rf_y_proba[:, 1])
tr_fpr, tr_tpr, tr_threasholds = roc_curve(y_test, tr_y_proba[:, 1])
Xgbc_fpr, Xgbc_tpr, Xgbc_threasholds = roc_curve(y_test, Xgbc_y_proba[:, 1])
svm_fpr, svm_tpr, svm_threasholds = roc_curve(y_test, svm_y_proba[:, 1])
knn_fpr, knn_tpr, knn_threasholds = roc_curve(y_test, knn_y_proba[:, 1])

roc_auc_lr = auc(lr_fpr, lr_tpr)
roc_auc_rf = auc(rf_fpr, rf_tpr)
roc_auc_tr = auc(tr_fpr, tr_tpr)
roc_auc_Xgbc = auc(Xgbc_fpr, Xgbc_tpr)
roc_auc_svm = auc(svm_fpr, svm_tpr)
roc_auc_knn = auc(knn_fpr, knn_tpr)
print(roc_auc_lr)
print(roc_auc_rf)
print(roc_auc_tr)
print(roc_auc_Xgbc)
print(roc_auc_knn)
print(roc_auc_svm)

    
def drawRoc(roc_auc,fpr,tpr):
    plt.subplots(figsize=(10, 7))
    plt.plot(lr_fpr, lr_tpr, color='#0000cc', lw=1.3, label='LR (AUC: {:.3f})'.format(roc_auc_lr))
    plt.plot(tr_fpr, tr_tpr, color='#ff0000', lw=1.3, label='DT (AUC: {:.3f})'.format(roc_auc_tr))
    plt.plot(Xgbc_fpr, Xgbc_tpr, color='#006633', lw=1.3, label='XGB (AUC: {:.3f})'.format(roc_auc_Xgbc))
    plt.plot(svm_fpr, svm_tpr, color='#000000', lw=1.3, label='SVM (AUC: {:.3f})'.format(roc_auc_svm))
    plt.plot(knn_fpr, knn_tpr, color='#663300', lw=1.3, label='KNN (AUC: {:.3f})'.format(roc_auc_knn))
    plt.plot(rf_fpr, rf_tpr, color='#ff0099', lw=1.3, label='RF (AUC: {:.3f})'.format(roc_auc_rf))
    plt.plot([0, 1], [0, 1], color='#999999', lw=1.3, linestyle='--', label='Random classifier')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('1-Specificity', fontsize=13)
    plt.ylabel('Sensitivity', fontsize=13)
    plt.title('Mortality-ROC Curve', fontsize=13)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('Mortality-ROC_300dpinew.pdf', dpi=300)
drawRoc(roc_auc_knn,knn_fpr,knn_tpr)



