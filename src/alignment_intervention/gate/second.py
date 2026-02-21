import os
import joblib
from pathlib import Path
import sys

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.classify import ValueAlignmentClassifier


class AlignmentInterventionLayer:
    """
    第二层
    使用训练好的分类器处理输入文本，返回属于负例的概率值
    """
    
    def __init__(self, model_save_dir=None):
        """
        初始化对齐干预层
        
        Args:
            model_save_dir (str): 模型保存目录，默认为项目中的标准位置
        """
        if model_save_dir is None:
            model_save_dir = os.path.join(project_root, "models", "value_alignment_classifiers")
        
        self.classifier = ValueAlignmentClassifier()
        self.model_save_dir = model_save_dir
        
        # 尝试加载已训练的模型
        try:
            self.classifier.load_models(self.model_save_dir)
            print(f"已成功加载模型，模型目录: {self.model_save_dir}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请先运行分类器训练脚本以生成模型")
    
    def get_negative_probability(self, text, value_type):
        """
        获取输入文本属于负例的概率值
        
        Args:
            text (str): 输入文本
            value_type (str): 价值观类型（如 'achievement', 'benevolence' 等）
            
        Returns:
            float: 属于负例的概率值（0-1之间）
        """
        try:
            # 使用加载的分类器预测负例概率
            probability = self.classifier.predict_negative_probability(text, value_type)
            return probability
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            return None
    
    def process_batch(self, texts, value_type):
        """
        批量处理文本
        
        Args:
            texts (list): 文本列表
            value_type (str): 价值观类型
            
        Returns:
            list: 每个文本对应的负例概率值列表
        """
        probabilities = []
        for text in texts:
            prob = self.get_negative_probability(text, value_type)
            probabilities.append(prob)
        return probabilities
    
    def get_supported_values(self):
        """
        获取支持的价值观类型列表
        
        Returns:
            list: 支持的价值观类型列表
        """
        return self.classifier.value_types


def main():

    # 创建实例
    intervention_layer = AlignmentInterventionLayer()
    
    # 显示支持的价值观类型
    print("支持的价值观类型:", intervention_layer.get_supported_values())
    
    # 测试文本
    test_text = "No matter how hard I work, I never feel like I have accomplished anything meaningful or worthwhile."
    
    # 对不同的价值观类型测试文本
    for value_type in intervention_layer.get_supported_values()[:3]:  # 只测试前3个类型
        prob = intervention_layer.get_negative_probability(test_text, value_type)
        if prob is not None:
            print(f"文本 '{test_text[:50]}...' 在 {value_type} 价值观下属于负例的概率: {prob:.4f}")
        else:
            print(f"无法计算 {value_type} 价值观下的概率")
    
    # 单独测试成就价值观（如果存在）
    if 'achievement' in intervention_layer.get_supported_values():
        prob = intervention_layer.get_negative_probability(test_text, 'achievement')
        if prob is not None:
            print(f"\n详细结果 - 成就价值观下文本属于负例的概率: {prob:.4f}")


if __name__ == "__main__":
    main()