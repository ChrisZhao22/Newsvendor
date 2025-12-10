import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取数据
df = pd.read_csv('../data/input_data/data_raw_by_hour.csv')

# 转换 datetime 列
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

# 2. 数据聚合：从“小时”变成“天”：实现颗粒度对齐
# 我们假设：
# - 需求 (count) 是当天的总和
# - 天气/温度 (temp, weather等) 取当天的平均值或最大值
daily_data = df.groupby('date').agg({
    'count': 'sum',  # 当天总需求
    'temp': 'mean',  # 平均气温
    'humidity': 'mean',  # 平均湿度
    'windspeed': 'mean',  # 平均风速
    'season': 'first',  # 季节 (取第一个值即可)
    'holiday': 'first',  # 是否节假日
    'workingday': 'first',  # 是否工作日
    'weather': lambda x: x.mode()[0]  # 天气状况 (取众数，即当天最常见的天气)
}).reset_index()

# 重命名 count 为 demand，方便理解
daily_data.rename(columns={'count': 'demand'}, inplace=True)


daily_data.to_csv('../data/demand_by_day.csv',index=False)

# 训练测试数据8:2
train_set, test_set = train_test_split(daily_data, test_size=0.2, random_state=42)

train_set.to_csv('../data/train_set.csv', index=False)
test_set.to_csv('../data/test_set.csv', index=False)

print(f"Original dataset size: {len(df)}")
print(f"Train set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")