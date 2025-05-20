import pandas as pd

def check_newlines_in_column(file_path, column_name):
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件时出现错误：{e}")
        return

    # 检查指定列是否存在换行符
    has_newlines = df[column_name].apply(lambda x: '\n' in str(x) or '\r' in str(x))

    # 输出带有换行符的行（如果存在）
    if has_newlines.any():
        print(f"列 '{column_name}' 中存在换行符的行如下：")
        print(df[has_newlines])
    else:
        print(f"列 '{column_name}' 没有包含换行符的行。")

# 使用示例
file_path = '/ossfs/workspace/dataset_v1/img_des/img_description_batch_2.csv'  # 替换为你的CSV文件路径
column_name = '动画图层描述'  # 替换为你要检查的列名

check_newlines_in_column(file_path, column_name)
