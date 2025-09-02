
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 混淆矩阵数据
confusion_matrix = np.array([[204, 64],
                             [89, 154]])

# 创建一个图形和子图
fig, ax = plt.subplots()

# 使用seaborn的热力图绘制混淆矩阵
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)

# 设置x轴标签
ax.set_xlabel('Prediction')
ax.set_xlabel('True')

# 设置y轴标签
ax.set_ylabel('True')
ax.set_ylabel('Prediction')

# 设置标题
ax.set_title('Mortality-Confusion matrix(RF)')

# 显示图形
plt.show()
