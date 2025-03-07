import pandas as pd
import numpy as np

# 读取正样本和负样本的CSV文件
positive_samples = pd.read_csv('./eval/positive_data.csv')
negative_samples = pd.read_csv('./eval/fake_data.csv')

# 设置要选择的样本数量
# n = 14741  # 你可以根据需要调整这个值
# n = len(positive_samples)


# 从正样本中随机选择n行
selected_positive = positive_samples.sample(n=n, random_state=42)

# 从负样本中随机选择n行
selected_negative = negative_samples.sample(n=n, random_state=42)

# 合并选择的正样本和负样本
combined_samples = pd.concat([selected_positive, selected_negative])

# 随机打乱合并后的数据
shuffled_samples = combined_samples.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存打乱后的数据到新的CSV文件
shuffled_samples.to_csv('./dataset_json/Final_data_process.csv', index=False)

print(f"已保存打乱后的 {2*n} 行样本到 'Final_data_process.csv'")