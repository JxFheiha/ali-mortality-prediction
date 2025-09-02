import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import _get_column_indices
# 读取.xls文件
#file_path = 'E:/ML/ALI-stroke/pythonML/ALI-stroke-preprocess.xlsx'
#data = pd.read_excel(file_path, engine='openpyxl')
data = pd.read_excel('E:/ML/ALI-stroke/pythonML/ALI-stroke-train_output.xlsx')

print(data.head())
print(data['Stroke'].value_counts())

# 假设 'stroke' 是目标变量，其它列是特征
X = data.drop(['Stroke', 'seqn'], axis=1)  # 特征，排除'Stroke'和'seqn'
y = data['Stroke']


# 应用SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 检查平衡后的数据
print("Resampled classes:")
print(pd.Series(y_resampled).value_counts())



# 将处理后的数据合并回DataFrame
resampled_data = pd.concat([X_resampled, y_resampled, data['seqn']], axis=1)

# 保存或进一步处理数据
resampled_data.to_excel('E:/ML/ALI-stroke/pythonML/Stroke-train-smote-data.xlsx', index=False)
