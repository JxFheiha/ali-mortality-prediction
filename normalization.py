import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 加载训练集数据
train_data = pd.read_excel('E:/ML/ALI-stroke/pythonML/ALI-stroke-train_output.xlsx')

# 分离特征和目标变量
X_train = train_data.drop(['Stroke', 'seqn'], axis=1)  # 特征
y_train = train_data['Stroke']  # 目标变量

# 定义特征归一化和one-hot编码的预处理器
# 假设分类变量为'Gender', 'Ethnicity', 'Education level', 'Marital status', 'Smoking status', 'Drinking status'
categorical_features = ['Gender', 'Ethnicity', 'Education level', 'Marital status', 'Smoking status', 'CVD','DM','Hypertension','Drinking status']
numerical_features = [col for col in X_train.columns if col not in categorical_features]

# 创建ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # 不处理剩余的列
)

# 创建Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# 对训练集应用Pipeline
X_train_transformed = pipeline.fit_transform(X_train)

# 获取数值特征和分类特征的列名
numerical_columns = numerical_features
categorical_columns = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features=categorical_features)

# 将处理后的数据转换为DataFrame
# 注意：OneHotEncoder的输出是一个numpy数组，需要转换为DataFrame并添加列名
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=numerical_columns + list(categorical_columns))

# 将目标变量与处理后的特征数据合并
train_transformed_data = pd.concat([X_train_transformed_df, y_train], axis=1)

# 输出处理后的数据到新的.xlsx文件
train_transformed_data.to_excel('E:/ML/ALI-stroke/pythonML/stroke-train-transformed-data.xlsx', index=False)
