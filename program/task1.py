import pandas as pd
import numpy as np
df = pd.read_excel(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386'
                   r'282450bb7c\非洲通讯产品销售数据.xlsx', sheet_name='SalesData')
# df.info()
'''
按年份分类
'''
df['quarter'] = df['日期'].dt.quarter
df['year'] = df['日期'].dt.year
# df.to_csv(r'D:\xinjianqq\比赛练习\1.csv')
data_country = df[['国家', '销售额', '利润', 'year']].groupby(['year', '国家']).agg('sum')
data_area = df[['地区', '销售额', '利润', 'year']].groupby(['year', '地区']).agg('sum')
data_server = df[['服务分类', '销售额', '利润', 'year']].groupby(['year', '服务分类']).agg('sum')
#   [52, 5, 3]  country ,area   server
def modef(X, i):
    X['销售额同比'] = X['销售额'].pct_change(periods=i)
    X['销售额同比'] = X['销售额同比'].apply(lambda x: str(round(x * 100, 2)) + '%').str.replace('nan%', '')
    X['利润同比'] = X['利润'].pct_change(periods=i)
    X['利润同比'] = X['利润同比'].apply(lambda x: str(round(x * 100, 2)) + '%').str.replace('nan%', '')
    return X

data_country = modef(data_country,52)
data_area = modef(data_area,5)
data_server = modef(data_server,3)


data_country.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\year_country.csv')
data_area.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\year_area.csv')
data_server.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\year_server.csv')

'''
按照季度分配
'''
n = [2017, 2018, 2019, 2020]
path = ['17_country.csv', '17_area.csv', '17_server.csv', '18_country.csv.csv', '18_area.csv', '18_server.csv'
        , '19_country.csv', '19_area.csv', '19_server.csv', '20_country.csv', '20_area.csv', '20_server.csv']
for i in range(len(n)):
    data_new = df[df['year'] == n[i]]           # 2017年的各季度数据
    data_new_country = data_new[['国家', '销售额', '利润', 'quarter']].groupby(['quarter', '国家']).agg('sum')
    data_new_area = data_new[['地区', '销售额', '利润', 'quarter']].groupby(['quarter', '地区']).agg('sum')
    data_new_server = data_new[['服务分类', '销售额', '利润', 'quarter']].groupby(['quarter', '服务分类']).agg('sum')
    data_new_country = modef(data_new_country, 52)
    data_new_area = modef(data_new_area, 5)
    data_new_server = modef(data_new_server, 3)
    i = 3*i
    data_new_country.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\20'+path[i])
    data_new_area.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\20'+path[i+1])
    data_new_server.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\20'+path[i+2])


"""
# task1.2
# """
data_1_2_area = df[['地区', '服务分类', '销售额', '利润']].groupby(['地区', '服务分类']).agg('sum')
data_1_2_area.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\data_1_2_area.csv')
data_1_2_country = df[['国家', '服务分类', '销售额', '利润']].groupby(['国家', '服务分类']).agg('sum')
data_1_2_country.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\data_1_2_country.csv')

"""
task1.3
"""
df = pd.read_excel(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386'
                   r'282450bb7c\非洲通讯产品销售数据.xlsx', sheet_name='SalespersonData')

da = df[['销售合同', '成交率', '销售经理']].groupby('销售经理').agg({'销售合同':np.sum,'成交率':np.mean})
da.to_csv(r'D:\xinjianqq\比赛练习\e9c455a98413d9a54ba386282450bb7c\result\销售经理.csv')
