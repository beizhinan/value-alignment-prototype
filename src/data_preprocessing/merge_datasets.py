import pandas as pd
import os
import glob
from pathlib import Path


def merge_value_datasets(controlled_dir, non_controlled_dir, output_dir):
    """
    合并相同价值观的controlled和non_controlled数据集
    
    Args:
        controlled_dir (str): controlled数据集目录
        non_controlled_dir (str): non_controlled数据集目录
        output_dir (str): 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有价值观名称（从controlled目录获取）
    controlled_files = glob.glob(os.path.join(controlled_dir, "*_context_controlled.csv"))
    
    # 提取价值观名称
    value_names = []
    for file in controlled_files:
        base_name = os.path.basename(file)
        value_name = base_name.replace("_context_controlled.csv", "")
        value_names.append(value_name)
    
    print(f"找到 {len(value_names)} 个价值观的数据集: {value_names}")
    
    for value_name in value_names:
        print(f"\n正在处理价值观: {value_name}")
        
        # 构建文件路径
        controlled_file = os.path.join(controlled_dir, f"{value_name}_context_controlled.csv")
        non_controlled_file = os.path.join(non_controlled_dir, f"{value_name}_non_context_controlled.csv")
        output_file = os.path.join(output_dir, f"{value_name}_merged.csv")
        
        # 检查文件是否存在
        if not os.path.exists(controlled_file):
            print(f"  警告: {controlled_file} 不存在，跳过")
            continue
        
        if not os.path.exists(non_controlled_file):
            print(f"  警告: {non_controlled_file} 不存在，跳过")
            continue
        
        # 读取两个数据集
        print(f"  读取 {os.path.basename(controlled_file)}...")
        df_controlled = pd.read_csv(controlled_file)
        
        print(f"  读取 {os.path.basename(non_controlled_file)}...")
        df_non_controlled = pd.read_csv(non_controlled_file)
        
        # 为了区分来源，可以添加一个标识列
        df_controlled['source'] = 'controlled'
        df_non_controlled['source'] = 'non_controlled'
        
        # 合并数据集
        print(f"  合并数据集...")
        merged_df = pd.concat([df_controlled, df_non_controlled], ignore_index=True)
        
        # 重新生成ID列（如果存在）
        if 'id' in merged_df.columns:
            merged_df['id'] = range(1, len(merged_df) + 1)
        
        # 保存合并后的数据集
        merged_df.to_csv(output_file, index=False)
        print(f"  已保存合并后的数据集到: {output_file}")
        print(f"  合并后数据集大小: {len(merged_df)} 行")
        print(f"  其中 {len(df_controlled)} 行来自controlled数据集")
        print(f"  {len(df_non_controlled)} 行来自non_controlled数据集")


def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 定义目录路径
    controlled_dir = os.path.join(project_root, "data", "controlled")
    non_controlled_dir = os.path.join(project_root, "data", "non_controlled")
    output_dir = os.path.join(project_root, "data", "merged")
    
    # 检查输入目录是否存在
    if not os.path.exists(controlled_dir):
        print(f"错误: 目录 {controlled_dir} 不存在")
        return
    
    if not os.path.exists(non_controlled_dir):
        print(f"错误: 目录 {non_controlled_dir} 不存在")
        return
    
    print("开始合并相同价值观的controlled和non_controlled数据集...")
    print("="*60)
    
    merge_value_datasets(controlled_dir, non_controlled_dir, output_dir)
    
    print("="*60)
    print("数据集合并完成!")


if __name__ == "__main__":
    main()