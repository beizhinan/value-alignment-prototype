import pandas as pd
import os
import glob


def add_id_to_csv(input_file_path):
    """
    为CSV文件的每一行添加一个整数ID，并直接覆盖原文件
    
    Args:
        input_file_path (str): 输入的CSV文件路径
    """
    # 读取CSV文件
    print(f"  正在读取文件: {os.path.basename(input_file_path)}")
    df = pd.read_csv(input_file_path)
    
    # 如果原文件已经有id列，则删除它，避免重复
    if 'id' in df.columns:
        print(f"  文件已包含ID列，将重新生成ID")
        df = df.drop(columns=['id'])
    
    # 在第一列插入ID列，ID从1开始递增
    df.insert(0, 'id', range(1, len(df) + 1))
    
    # 直接覆盖原文件
    df.to_csv(input_file_path, index=False)
    
    print(f"  已成功为 {os.path.basename(input_file_path)} 添加ID列并覆盖原文件")
    print(f"  文件包含 {len(df)} 行数据，ID范围: 1 - {len(df)}")
    return len(df)


def process_all_csv_files(data_dir):
    """
    处理data目录下的所有CSV文件
    
    Args:
        data_dir (str): 数据目录路径
    """
    print(f"开始扫描目录: {data_dir}")
    
    # 使用glob模式查找所有CSV文件
    csv_pattern = os.path.join(data_dir, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    print(f"找到 {len(csv_files)} 个CSV文件需要处理")
    
    # 按目录分组显示文件
    directories = {}
    for file_path in csv_files:
        dir_path = os.path.dirname(file_path)
        if dir_path not in directories:
            directories[dir_path] = []
        directories[dir_path].append(file_path)
    
    print("\n文件分布:")
    for dir_path, files in directories.items():
        print(f"  {dir_path}: {len(files)} 个文件")
        for file in files:
            print(f"    - {os.path.basename(file)}")
    
    total_rows = 0
    processed_files = 0
    
    print("\n开始处理文件...")
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"\n[{i}/{len(csv_files)}] 正在处理: {os.path.basename(csv_file)}")
            rows = add_id_to_csv(csv_file)
            total_rows += rows
            processed_files += 1
            print(f"  完成处理 ({processed_files}/{len(csv_files)})")
        except Exception as e:
            print(f"  处理文件 {csv_file} 时出错: {str(e)}")
    
    print(f"\n处理完成!")
    print(f"总共处理了 {processed_files} 个文件")
    print(f"总共处理了 {total_rows} 行数据")


def main():
    # 获取项目根目录下的data文件夹路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：找不到目录 {data_dir}")
        return
    
    print("开始为CSV文件添加ID...")
    print("="*60)
    
    # 处理data目录下的所有CSV文件
    process_all_csv_files(data_dir)
    
    print("="*60)
    print("所有CSV文件处理完毕!")


if __name__ == "__main__":
    main()