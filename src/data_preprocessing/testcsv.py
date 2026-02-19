import os
import pandas as pd
import glob

def txt_to_csv(txt_file_path, csv_file_path=None):
    """
    将单个txt文件转换为csv格式
    
    参数:
    txt_file_path: 输入txt文件路径
    csv_file_path: 输出csv文件路径，如果为None，则使用txt文件同名的csv文件
    """
    if csv_file_path is None:
        csv_file_path = txt_file_path.replace('.txt', '.csv')
    
    # 读取txt文件的每一行
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 清理每一行（去除首尾空白字符）
    lines = [line.strip() for line in lines if line.strip()]
    
    # 创建DataFrame，不指定列名
    df = pd.DataFrame(lines)
    
    # 保存为CSV文件，不包含列标题和索引
    df.to_csv(csv_file_path, index=True, header=False)
    
    print(f"已将 {txt_file_path} 转换为 {csv_file_path}")
    print(f"共处理了 {len(lines)} 行数据")


def convert_all_txt_to_csv(root_dir):
    """
    转换指定目录及其子目录下所有txt文件为csv格式
    
    参数:
    root_dir: 根目录路径
    """
    # 查找所有txt文件
    txt_files = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)
    
    if not txt_files:
        print(f"在 {root_dir} 目录及其子目录中没有找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件需要转换")
    
    for i, txt_file in enumerate(txt_files, 1):
        # 生成对应的csv文件路径
        csv_file = txt_file.replace('.txt', '.csv')
        
        print(f"[{i}/{len(txt_files)}] 正在转换: {txt_file}")
        
        try:
            txt_to_csv(txt_file, csv_file)
        except Exception as e:
            print(f"转换 {txt_file} 时出错: {str(e)}")
    
    print("转换完成！")


if __name__ == "__main__":
    # 获取项目根目录的路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = os.path.join(project_root, "data", "test")
    
    # 检查test目录是否存在
    if not os.path.exists(test_dir):
        print(f"目录 {test_dir} 不存在")
    else:
        # 只转换test目录下的txt文件
        print(f"开始转换 {test_dir} 目录下的所有txt文件...")
        convert_all_txt_to_csv(test_dir)