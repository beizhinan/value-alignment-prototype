import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import sys
import torch
from typing import Tuple, Optional

from src.models.model import T5Model

def calculate_text_prototype_similarity(
    input_text: str, 
    model, 
    value_type: str,
    prototype_base_dir: str = None,
    threshold: float = 0.0
) -> Tuple[float, str, Optional[str]]:
    """
    计算输入文本与原型库向量之间的相似度
    
    Args:
        input_text: 输入的文本
        model: 预加载的模型，用于提取文本向量
        value_type: 输入文本所属的价值观类型（如 'achievement', 'benevolence' 等）
        prototype_base_dir: 原型库基础目录路径，默认为项目原型库目录
        threshold: 相似度阈值，低于此值则返回None
    
    Returns:
        Tuple[float, str, Optional[str]]: (最大相似度值, 最相似原型类型(positive/negative), 最相似原型文本)
    """
    
    # 如果没有指定原型库基础目录，则使用默认路径
    if prototype_base_dir is None:
        prototype_base_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'prototype_learning', 'prototypes'
        )
    
    # 构建该价值观类型的原型库JSON文件路径
    prototype_json_path = os.path.join(prototype_base_dir, value_type, f'{value_type}_prototype_data.json')
    
    # 检查文件是否存在
    if not os.path.exists(prototype_json_path):
        raise FileNotFoundError(f"原型库文件不存在: {prototype_json_path}")
    
    # 加载原型库数据
    with open(prototype_json_path, 'r', encoding='utf-8') as f:
        prototype_data = json.load(f)
    
    # 提取正负原型向量
    positive_prototype_vectors = np.array(prototype_data['positive_prototype_vectors'])
    negative_prototype_vectors = np.array(prototype_data['negative_prototype_vectors'])
    
    # 提取代表性文本（如果有）
    positive_texts = prototype_data.get('positive_prototype_texts', {})
    negative_texts = prototype_data.get('negative_prototype_texts', {})
    
    # 提取输入文本的向量表示
    input_vector = extract_text_vector(input_text, model)
    input_vector = input_vector.reshape(1, -1)  # reshape为sklearn所需的格式
    
    # 计算与正原型的相似度
    if len(positive_prototype_vectors) > 0:
        positive_similarities = cosine_similarity(input_vector, positive_prototype_vectors)
        max_positive_sim = np.max(positive_similarities)
        max_positive_idx = np.argmax(positive_similarities)
        
        # 获取最相似正原型的文本描述
        pos_cluster_key = f'cluster_{max_positive_idx}'
        positive_text = positive_texts.get(pos_cluster_key, "Unknown prototype text")
    else:
        max_positive_sim = -float('inf')
        positive_text = None
    
    # 计算与负原型的相似度
    if len(negative_prototype_vectors) > 0:
        negative_similarities = cosine_similarity(input_vector, negative_prototype_vectors)
        max_negative_sim = np.max(negative_similarities)
        max_negative_idx = np.argmax(negative_similarities)
        
        # 获取最相似负原型的文本描述
        neg_cluster_key = f'cluster_{max_negative_idx}'
        negative_text = negative_texts.get(neg_cluster_key, "Unknown prototype text")
    else:
        max_negative_sim = -float('inf')
        negative_text = None
    
    # 判断最大相似度是哪个原型
    if max_positive_sim >= max_negative_sim:
        similarity_value = max_positive_sim
        prototype_type = "positive"
        prototype_text = positive_text
    else:
        similarity_value = -max_negative_sim  # 负原型相似度为负值
        prototype_type = "negative"
        prototype_text = negative_text
    
    # 应用阈值过滤
    if abs(similarity_value) < threshold:
        return 0.0, "neutral", None
    
    return similarity_value, prototype_type, prototype_text


def extract_text_vector(text: str, model) -> np.ndarray:
    """
    使用模型提取文本向量
    
    Args:
        text: 输入文本
        model: 预加载的模型
        
    Returns:
        np.ndarray: 文本的向量表示
    """
    # 获取编码器最后一层的隐藏状态
    last_hidden_state = model.get_encoder_last_hidden_state(text)
    
    # 对序列维度进行平均池化，得到固定长度的向量
    pooled_vector = torch.mean(last_hidden_state, dim=1).squeeze(0)
    
    # 转换为numpy数组
    vector = pooled_vector.detach().cpu().numpy()
    
    return vector


def find_closest_prototype(
    input_text: str, 
    model, 
    value_type: str,
    prototype_base_dir: str = None
) -> Tuple[str, float, str]:
    """
    找到与输入文本最相似的原型，返回原型类型、相似度和文本内容
    
    Args:
        input_text: 输入的文本
        model: 预加载的模型
        value_type: 输入文本所属的价值观类型
        prototype_base_dir: 原型库基础目录路径，默认为项目原型库目录
    
    Returns:
        Tuple[str, float, str]: (原型类型, 相似度值, 原型文本)
    """
    similarity_value, prototype_type, prototype_text = calculate_text_prototype_similarity(
        input_text, model, value_type, prototype_base_dir
    )
    
    return prototype_type, abs(similarity_value), prototype_text


def get_all_prototype_similarities(
    input_text: str, 
    model, 
    value_type: str,
    prototype_base_dir: str = None
) -> Tuple[list, list]:
    """
    计算输入文本与所有正负原型的相似度
    
    Args:
        input_text: 输入的文本
        model: 预加载的模型
        value_type: 输入文本所属的价值观类型
        prototype_base_dir: 原型库基础目录路径，默认为项目原型库目录
    
    Returns:
        Tuple[list, list]: (正原型相似度列表, 负原型相似度列表)
    """
    # 如果没有指定原型库基础目录，则使用默认路径
    if prototype_base_dir is None:
        prototype_base_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'prototype_learning', 'prototypes'
        )
    
    # 构建该价值观类型的原型库JSON文件路径
    prototype_json_path = os.path.join(prototype_base_dir, value_type, f'{value_type}_prototype_data.json')
    
    # 检查文件是否存在
    if not os.path.exists(prototype_json_path):
        raise FileNotFoundError(f"原型库文件不存在: {prototype_json_path}")
    
    # 加载原型库数据
    with open(prototype_json_path, 'r', encoding='utf-8') as f:
        prototype_data = json.load(f)
    
    # 提取正负原型向量
    positive_prototype_vectors = np.array(prototype_data['positive_prototype_vectors'])
    negative_prototype_vectors = np.array(prototype_data['negative_prototype_vectors'])
    
    # 提取输入文本的向量表示
    input_vector = extract_text_vector(input_text, model)
    input_vector = input_vector.reshape(1, -1)
    
    # 计算与所有正原型的相似度
    if len(positive_prototype_vectors) > 0:
        positive_similarities = cosine_similarity(input_vector, positive_prototype_vectors)[0]
    else:
        positive_similarities = []
    
    # 计算与所有负原型的相似度
    if len(negative_prototype_vectors) > 0:
        negative_similarities = cosine_similarity(input_vector, negative_prototype_vectors)[0]
    else:
        negative_similarities = []
    
    return list(positive_similarities), list(negative_similarities)


def find_best_matching_prototype_type_and_value(
    input_text: str, 
    model, 
    value_type: str,
    prototype_base_dir: str = None
) -> Tuple[str, float]:
    """
    仅返回最匹配的原型类型和相似度值（不返回文本），便于集成到其他系统中
    
    Args:
        input_text: 输入的文本
        model: 预加载的模型
        value_type: 输入文本所属的价值观类型
        prototype_base_dir: 原型库基础目录路径，默认为项目原型库目录
    
    Returns:
        Tuple[str, float]: (原型类型, 相似度值)
    """
    similarity_value, prototype_type, _ = calculate_text_prototype_similarity(
        input_text, model, value_type, prototype_base_dir
    )
    
    return prototype_type, abs(similarity_value)


def find_prototype_with_metadata(
    input_text: str, 
    model, 
    value_type: str,
    prototype_base_dir: str = None
) -> dict:
    """
    返回完整的匹配结果，包含元数据
    
    Args:
        input_text: 输入的文本
        model: 预加载的模型
        value_type: 输入文本所属的价值观类型
        prototype_base_dir: 原型库基础目录路径，默认为项目原型库目录
    
    Returns:
        dict: 包含所有相关信息的字典
    """
    similarity_value, prototype_type, prototype_text = calculate_text_prototype_similarity(
        input_text, model, value_type, prototype_base_dir
    )
    
    result = {
        'input_text': input_text,
        'similarity_value': similarity_value,
        'prototype_type': prototype_type,
        'prototype_text': prototype_text,
        'absolute_similarity': abs(similarity_value),
        'match_confidence': abs(similarity_value),  # 相似度绝对值作为置信度
        'value_type': value_type  # 添加价值观类型到结果中
    }
    
    return result


def batch_calculate_similarity(
    input_texts: list, 
    model, 
    value_type: str,
    prototype_base_dir: str = None
) -> list:
    """
    批量计算多个文本与原型库的相似度
    
    Args:
        input_texts: 文本列表
        model: 预加载的模型
        value_type: 输入文本所属的价值观类型
        prototype_base_dir: 原型库基础目录路径，默认为项目原型库目录
    
    Returns:
        list: 每个文本的相似度结果列表
    """
    results = []
    for text in input_texts:
        similarity_value, prototype_type, prototype_text = calculate_text_prototype_similarity(
            text, model, value_type, prototype_base_dir
        )
        results.append({
            'text': text,
            'similarity_value': similarity_value,
            'prototype_type': prototype_type,
            'prototype_text': prototype_text,
            'absolute_similarity': abs(similarity_value),
            'value_type': value_type
        })
    
    return results

def example_usage():
    """
    示例：如何使用文本与原型库相似度计算功能
    """
    # 加载预训练模型（示例）
    print("正在加载模型...")
    try:
        model = T5Model(model_name='t5-3b') 
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保模型和依赖项已正确安装")
        return

    # 示例价值观类型
    value_type = "achievement"

    # 示例文本
    sample_text = "No matter how hard I work, I never feel like I have accomplished anything meaningful or worthwhile."

    print(f"\n输入文本: {sample_text}")
    print(f"价值观类型: {value_type}")

    # 1. 计算文本与原型库的相似度
    print("\n1. 计算文本与原型库的相似度:")
    try:
        similarity_value, prototype_type, prototype_text = calculate_text_prototype_similarity(
            sample_text, model, value_type
        )
        print(f"   相似度值: {similarity_value:.4f}")
        print(f"   最相似原型类型: {prototype_type}")
        print(f"   最相似原型文本: {prototype_text}")
    except FileNotFoundError as e:
        print(f"   错误: {e}")


if __name__ == "__main__":
    example_usage()