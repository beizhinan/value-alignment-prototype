import os
import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.alignment_intervention.gate.first import find_best_matching_prototype_type_and_value
from src.alignment_intervention.gate.second import AlignmentInterventionLayer
from src.models.model import T5Model


class GateController:
    """
    门控控制器
    """
    
    def __init__(self):
        """
        初始化门控控制器
        """
        # 加载T5模型用于第一层
        try:
            self.t5_model = T5Model()
            print("T5模型加载成功")
        except Exception as e:
            print(f"T5模型加载失败: {e}")
            self.t5_model = None
        
        # 初始化第二层干预系统
        self.second_layer = AlignmentInterventionLayer()
        
    def process_text(self, input_text, value_type, weight_first_layer=0.3, weight_second_layer=0.7):
        """
        处理输入文本，返回最终的对齐评分
        
        Args:
            input_text (str): 输入文本
            value_type (str): 价值观类型（如 'achievement', 'benevolence' 等）
            weight_first_layer (float): 第一层权重
            weight_second_layer (float): 第二层权重
            
        Returns:
            dict: 包含处理结果的字典
        """
        if self.t5_model is None:
            return {
                'error': 'T5模型未加载，无法执行第一层处理',
                'final_score': None
            }
        
        # 第一层处理：原型相似度检测
        prototype_type, similarity_value = find_best_matching_prototype_type_and_value(
            input_text, self.t5_model, value_type
        )
        
        print(f"第一层结果 - 原型类型: {prototype_type}, 相似度: {similarity_value:.4f}")
        
        # 根据第一层结果决定是否跳过第二层
        if prototype_type == "positive":
            # 如果是正原型，跳过第二层，直接返回正值
            final_score = similarity_value * weight_first_layer
            print(f"检测到正原型，跳过第二层，加权得分: {final_score:.4f}")
            
            return {
                'first_layer_result': {
                    'prototype_type': prototype_type,
                    'similarity_value': similarity_value
                },
                'second_layer_result': None,
                'final_score': final_score,
                'reason': 'positive_prototype_detected'
            }
        else:
            # 如果是负原型，进入第二层获取负例概率
            negative_probability = self.second_layer.get_negative_probability(input_text, value_type)
            
            if negative_probability is None:
                return {
                    'error': f'第二层处理失败，无法计算{value_type}价值观下的负例概率',
                    'final_score': None
                }
            
            print(f"第二层结果 - 负例概率: {negative_probability:.4f}")
            
            # 加权合并两层结果
            # 第一层贡献：相似度值 * 权重 * (-1 因为是负原型)
            first_contribution = -similarity_value * weight_first_layer
            # 第二层贡献：负例概率 * 权重
            second_contribution = negative_probability * weight_second_layer
            
            final_score = first_contribution + second_contribution
            
            print(f"第一层贡献: {first_contribution:.4f}")
            print(f"第二层贡献: {second_contribution:.4f}")
            print(f"最终得分: {final_score:.4f}")
            
            return {
                'first_layer_result': {
                    'prototype_type': prototype_type,
                    'similarity_value': similarity_value
                },
                'second_layer_result': {
                    'negative_probability': negative_probability
                },
                'final_score': final_score,
                'reason': 'negative_prototype_processed_by_both_layers'
            }


def main():
    """
    主函数，演示门控控制器的使用
    """
    print("初始化门控控制器...")
    controller = GateController()
    
    # 测试文本
    test_texts = [
        "I wake up every morning driven by the desire to push my limits, achieve something meaningful, and prove to myself that I am capable of greatness.",
        "No matter how hard I work, I never feel like I have accomplished anything meaningful or worthwhile."
    ]
    
    # 测试价值观类型
    value_types = ["achievement"]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"处理文本: {text[:60]}...")
        
        for value_type in value_types:
            print(f"\n价值观类型: {value_type}")
            
            result = controller.process_text(text, value_type)
            
            if 'error' in result:
                print(f"处理出错: {result['error']}")
            else:
                print(f"最终得分: {result['final_score']:.4f}")
                print(f"处理原因: {result['reason']}")
                
                if result['second_layer_result'] is not None:
                    print(f"- 第一层相似度: {result['first_layer_result']['similarity_value']:.4f}")
                    print(f"- 第二层负例概率: {result['second_layer_result']['negative_probability']:.4f}")


if __name__ == "__main__":
    main()