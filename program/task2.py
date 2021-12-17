import pandas as pd
from task1 import modef
df = pd.read_excel(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386'
                   r'282450bb7c\非洲通讯产品销售数据.xlsx', sheet_name='SalesData')
"""
国家
"""
df['year']=df['日期'].dt.year
df['quarter']= df['日期'].dt.quarter
data = df[['国家', '销售额', '利润', '服务分类', 'year', 'quarter']].groupby(['year', 'quarter','国家', '服务分类']).agg('sum')
data = modef(data, 213)

data.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\test.csv')

data_year = df[['国家', '销售额', '利润', '服务分类', 'year']].groupby(['year','国家','服务分类']).agg('sum')
data_year['销售年增长率'] = data_year['销售额'].pct_change(periods=130).apply(lambda x: str(round(x * 100, 2)) + '%').str.replace('nan%', '')
data_year['利润的年增长'] = data_year['利润'].pct_change(periods=130).apply(lambda x: str(round(x * 100, 2)) + '%').str.replace('nan%', '')

data_year.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\test1.csv')


"""
地区
"""

data1 = df[['销售额', '利润','地区', '服务分类', 'year', 'quarter']].groupby(['year', 'quarter','地区', '服务分类']).agg('sum')
data1 = modef(data1, 54)

data1.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\area_test.csv')

data_year1 = df[['地区','销售额', '利润', '服务分类', 'year']].groupby(['year','地区','服务分类']).agg('sum')
data_year1['销售年增长率'] = data_year1['销售额'].pct_change(periods=15).apply(lambda x: str(round(x * 100, 2)) + '%').str.replace('nan%', '')
data_year1['利润的年增长'] = data_year1['利润'].pct_change(periods=15).apply(lambda x: str(round(x * 100, 2)) + '%').str.replace('nan%', '')

data_year1.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\area_test1.csv')


"""
task2.3    国家
"""
import pandas as pd

# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

data=pd.read_excel('./data/非洲通讯产品销售数据.xlsx')

def yuce(data, str1):
    data = data.loc[data['国家'] == str1, :]
    data1 = data[['日期', '销售额']]
    print(len(data1))
    train = data1[0:15]
    test = data1[15:]

    # Aggregating the dataset at daily level
    data1['Timestamp'] = pd.to_datetime(data1['日期'], format='%d-%m-%Y %H:%M')  # 4位年用Y，2位年用y
    data1.index = data1['Timestamp']
    data1 = data1.resample('Q').mean()  # 按天采样，计算均值

    train['Timestamp'] = pd.to_datetime(train['日期'], format='%d-%m-%Y %H:%M')
    train.index = train['Timestamp']
    train = train.resample('Q').mean()  #

    test['Timestamp'] = pd.to_datetime(test['日期'], format='%d-%m-%Y %H:%M')
    test.index = test['Timestamp']
    test = test.resample('Q').mean()

    riqi = ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07',
            '2021-01-08', '2021-01-09', '2021-01-10', '2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14',
            '2021-01-15', '2021-01-16', '2021-01-17', '2021-01-18', '2021-01-19', '2021-01-20', '2021-01-21',
            '2021-01-22', '2021-01-23', '2021-01-24', '2021-01-25', '2021-01-26', '2021-01-27', '2021-01-28',
            '2021-01-29', '2021-01-30', '2021-01-31', '2021-02-01', '2021-02-02', '2021-02-03', '2021-02-04',
            '2021-02-05', '2021-02-06', '2021-02-07', '2021-02-08', '2021-02-09', '2021-02-10', '2021-02-11',
            '2021-02-12', '2021-02-13', '2021-02-14', '2021-02-15', '2021-02-16', '2021-02-17', '2021-02-18',
            '2021-02-19', '2021-02-20', '2021-02-21', '2021-02-22', '2021-02-23', '2021-02-24', '2021-02-25',
            '2021-02-26', '2021-02-27', '2021-02-28', '2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04',
            '2021-03-05', '2021-03-06', '2021-03-07', '2021-03-08', '2021-03-09', '2021-03-10', '2021-03-11',
            '2021-03-12', '2021-03-13', '2021-03-14', '2021-03-15', '2021-03-16', '2021-03-17', '2021-03-18',
            '2021-03-19', '2021-03-20', '2021-03-21', '2021-03-22', '2021-03-23', '2021-03-24', '2021-03-25',
            '2021-03-26', '2021-03-27', '2021-03-28', '2021-03-29', '2021-03-30', '2021-03-31']

    dfyuce = pd.DataFrame(riqi, columns=['日期'])
    # print(dfyuce)

    from statsmodels.tsa.api import Holt

    fit = Holt(np.asarray(train['销售额'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    dfyuce['销售额'] = fit.forecast(len(dfyuce))

    data2 = data[['日期', '利润']]
    print(len(data2))
    train2 = data2[0:200]
    test2 = data2[200:]
    # Aggregating the dataset at daily level
    data2['Timestamp'] = pd.to_datetime(data2['日期'], format='%d-%m-%Y %H:%M')  # 4位年用Y，2位年用y
    data2.index = data2['Timestamp']
    data2 = data2.resample('Q').mean()  # 按天采样，计算均值


    train2['Timestamp'] = pd.to_datetime(train2['日期'], format='%d-%m-%Y %H:%M')
    train2.index = train2['Timestamp']
    train2 = train2.resample('Q').mean()  #

    test2['Timestamp'] = pd.to_datetime(test2['日期'], format='%d-%m-%Y %H:%M')
    test2.index = test2['Timestamp']
    test2 = test2.resample('Q').mean()


    from statsmodels.tsa.api import Holt


    fit = Holt(np.asarray(train2['利润'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    dfyuce['利润'] = fit.forecast(len(dfyuce))

    return dfyuce

# print(dfyuce)


# print(data['地区'])
# print(set(data['地区']))
# print(set(data['国家']))

diqu = list(set(data['地区']))
guojia = list(set(data['国家']))
df = pd.DataFrame(columns=['日期', '销售额', '利润', '国家'])
for co in guojia:
    a = co

    df_co = yuce(data, co)
    df_co['国家'] = co
    # print(df_co)
    df = pd.concat([df, df_co])

# print(df)

df.to_csv('./data/国家预测销售额及利润.csv')


"""
task  2.3 地区
"""
import pandas as pd

# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

data=pd.read_excel('./data/非洲通讯产品销售数据.xlsx')

def yuce(data, str1):
    data = data.loc[data['地区'] == str1, :]
    data1 = data[['日期', '销售额']]
    print(len(data1))
    train = data1[0:200]
    test = data1[200:]

    # Aggregating the dataset at daily level
    data1['Timestamp'] = pd.to_datetime(data1['日期'], format='%d-%m-%Y %H:%M')  # 4位年用Y，2位年用y
    data1.index = data1['Timestamp']
    data1 = data1.resample('Q').mean()  # 按天采样，计算均值

    train['Timestamp'] = pd.to_datetime(train['日期'], format='%d-%m-%Y %H:%M')
    train.index = train['Timestamp']
    train = train.resample('Q').mean()  #

    test['Timestamp'] = pd.to_datetime(test['日期'], format='%d-%m-%Y %H:%M')
    test.index = test['Timestamp']
    test = test.resample('Q').mean()

    riqi = ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07',
            '2021-01-08', '2021-01-09', '2021-01-10', '2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14',
            '2021-01-15', '2021-01-16', '2021-01-17', '2021-01-18', '2021-01-19', '2021-01-20', '2021-01-21',
            '2021-01-22', '2021-01-23', '2021-01-24', '2021-01-25', '2021-01-26', '2021-01-27', '2021-01-28',
            '2021-01-29', '2021-01-30', '2021-01-31', '2021-02-01', '2021-02-02', '2021-02-03', '2021-02-04',
            '2021-02-05', '2021-02-06', '2021-02-07', '2021-02-08', '2021-02-09', '2021-02-10', '2021-02-11',
            '2021-02-12', '2021-02-13', '2021-02-14', '2021-02-15', '2021-02-16', '2021-02-17', '2021-02-18',
            '2021-02-19', '2021-02-20', '2021-02-21', '2021-02-22', '2021-02-23', '2021-02-24', '2021-02-25',
            '2021-02-26', '2021-02-27', '2021-02-28', '2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04',
            '2021-03-05', '2021-03-06', '2021-03-07', '2021-03-08', '2021-03-09', '2021-03-10', '2021-03-11',
            '2021-03-12', '2021-03-13', '2021-03-14', '2021-03-15', '2021-03-16', '2021-03-17', '2021-03-18',
            '2021-03-19', '2021-03-20', '2021-03-21', '2021-03-22', '2021-03-23', '2021-03-24', '2021-03-25',
            '2021-03-26', '2021-03-27', '2021-03-28', '2021-03-29', '2021-03-30', '2021-03-31']

    dfyuce = pd.DataFrame(riqi, columns=['日期'])
    # print(dfyuce)

    from statsmodels.tsa.api import Holt

    fit = Holt(np.asarray(train['销售额'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    dfyuce['销售额'] = fit.forecast(len(dfyuce))

    data2 = data[['日期', '利润']]
    print(len(data2))
    train2 = data2[0:200]
    test2 = data2[200:]
    # Aggregating the dataset at daily level
    data2['Timestamp'] = pd.to_datetime(data2['日期'], format='%d-%m-%Y %H:%M')  # 4位年用Y，2位年用y
    data2.index = data2['Timestamp']
    data2 = data2.resample('Q').mean()  # 按天采样，计算均值


    train2['Timestamp'] = pd.to_datetime(train2['日期'], format='%d-%m-%Y %H:%M')
    train2.index = train2['Timestamp']
    train2 = train2.resample('Q').mean()  #

    test2['Timestamp'] = pd.to_datetime(test2['日期'], format='%d-%m-%Y %H:%M')
    test2.index = test2['Timestamp']
    test2 = test2.resample('Q').mean()


    from statsmodels.tsa.api import Holt


    fit = Holt(np.asarray(train2['利润'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    dfyuce['利润'] = fit.forecast(len(dfyuce))

    return dfyuce

# print(dfyuce)

df_Western = yuce(data, 'Western')
# print(df_Western)

# print(data['地区'])
# print(set(data['地区']))
# print(set(data['国家']))

diqu = list(set(data['地区']))
guojia = list(set(data['国家']))
df = pd.DataFrame(columns=['日期', '销售额', '利润', '地区'])
for co in diqu:
    a = co

    df_co = yuce(data, co)
    df_co['地区'] = co
    # print(df_co)
    df = pd.concat([df, df_co])

# print(df)

df.to_csv('./data/地区预测销售额及利润.csv')
