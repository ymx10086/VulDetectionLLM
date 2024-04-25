import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('GPT4.csv')

# 获取其中label列的数据
target_response = data['Label']

# 将其中Score列里面分数为1.5的更改为0
data['Score'] = data['Score'].apply(lambda x: 0 if x == 1.5 else x)

# 处理label列的数据，示例为C.CWE-377:Insecure Temporary File|A.CWE-668:Ex... 
# 通过正则表达式提取其中的数字
pattern = re.compile(r"CWE[-|:| ]?\s?(\d{1,3})")
target_response = target_response.apply(lambda x: pattern.findall(x))

# 将target_response分成两列
target_response = target_response.apply(lambda x: pd.Series(x))

# 将target_response的列名改为label1和label2
target_response.columns = ['label1', 'label2']
# 将target_response与data合并
data = pd.concat([data, target_response], axis=1)

# 取出其中第一列数目前20的类别，赋值为label，同时获取其数量，保存为dict
label1 = target_response['label1'].value_counts().head(20).to_dict()

# 取出其中第二列数目前20的类别，赋值为label，同时获取其数量，保存为dict
label2 = target_response['label2'].value_counts().head(20).to_dict()

# 输出label1和label2
print(label1)
print(label2)

# 获取data中label1列为label1中的值对应的Score的,并将key，value，value/label1[key]构建为dict
dict1 = {}
for key, value in label1.items():
    print(key, data[data['label1'] == key]['Score'].sum())
    print(data[data['label1'] == key]['Score'].sum() / value)
    dict1[key] = data[data['label1'] == key]['Score'].sum() / value

# 导出dict1
print(dict1)

# 将dict绘制成柱状图，并保存结果到本地
plt.bar(dict1.keys(), dict1.values())
plt.xticks(rotation=90)
plt.savefig('GPT4-1-keysort.eps')
plt.show()

# 将dict按照value排序
dict1 = dict(sorted(dict1.items(), key=lambda x: x[1], reverse=True))

# 将dict绘制成柱状图，并保存结果到本地
plt.bar(dict1.keys(), dict1.values())
plt.xticks(rotation=90)
plt.savefig('GPT4-1-valuesort.eps')
plt.show()



