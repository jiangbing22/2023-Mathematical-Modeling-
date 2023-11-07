import pandas as pd

def mean_normalize(data):
    normalized_data = (data - data.mean()) / data.std()
    return normalized_data

def preprocess(file):
    df = pd.read_excel(file)
    # 如前一样删除列
    df.drop(["证券代码", '公司发布财报的日期', '财报统计的季度的最后一天', '年份', '季度', '已获利息倍数', '主营营业收入(元)', '流通股本', '总股本', '股票名称', '流通市值', '所处行业', '市盈率(动)', '市净率', 'ROE', '毛利率', '净利率', '板块编号', '实际控制人名称', '控股数量(万股)', '控股比例(%)', '直接控制人名称', '控制方式'], axis=1, inplace=True)
    df.drop(df.columns[-1:-15:-1], axis=1, inplace=True)
    # 对数据框执行均值归一化
    df = mean_normalize(df)
    df.dropna(inplace=True)
    array=df.values
    #df.to_excel('result.xlsx')
    return array

if __name__ == "__main__":
    preprocess("000006.xlsx")
