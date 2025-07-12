from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# 获取训练和测试数据
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

# 保存数据到CSV文件
# 注意：20newsgroups数据集包含文本数据，而非特征矩阵
df_train = pd.DataFrame({
    'text': train_data.data,
    'target': train_data.target
})
df_train.to_csv('train.csv', index=False)

df_test = pd.DataFrame({
    'text': test_data.data, 
    'target': test_data.target
})
df_test.to_csv('test.csv', index=False)

print("数据集已成功保存为 train.csv 和 test.csv")
print(f"训练集大小: {len(df_train)}")
print(f"测试集大小: {len(df_test)}")
print(f"类别数量: {len(train_data.target_names)}")