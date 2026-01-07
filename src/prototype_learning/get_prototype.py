import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from typing import List, Tuple, Dict
import os
import warnings
warnings.filterwarnings('ignore')

from src.models.model import T5Model


class PrototypeLearner:
    """
    原型学习类，用于构建价值观原型向量库
    """
    
    def __init__(self, model: T5Model, csv_path: str):
        """
        初始化原型学习器
        
        Args:
            model: 已加载的T5模型
            csv_path: 包含正负样本的CSV文件路径
        """
        self.model = model
        self.csv_path = csv_path
        self.data = None
        self.positive_vectors = None
        self.negative_vectors = None
        self.prototypes = {}
        
        # 读取CSV数据
        self._load_data()
        
    def _load_data(self):
        """
        加载CSV数据
        """
        print(f"正在加载数据: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        
        # 检查数据结构
        if 'question_1' not in self.data.columns or 'question_2' not in self.data.columns:
            raise ValueError("CSV文件必须包含'question_1'和'question_2'列")
        
        print(f"数据加载完成，共{len(self.data)}条样本")
    
    def extract_vectors(self, texts: List[str]) -> np.ndarray:
        """
        使用T5模型提取文本向量
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 文本向量矩阵
        """
        print(f"正在提取{len(texts)}个文本的向量表示...")
        
        vectors = []
        for i, text in enumerate(texts):
            if i % 50 == 0:  # 每50个文本打印一次进度
                print(f"处理进度: {i}/{len(texts)}")
                
            # 获取编码器最后一层的隐藏状态
            last_hidden_state = self.model.get_encoder_last_hidden_state(text)
            
            # 对序列维度进行平均池化，得到固定长度的向量
            pooled_vector = torch.mean(last_hidden_state, dim=1).squeeze(0)
            
            # 转换为numpy数组
            vector = pooled_vector.detach().cpu().numpy()
            vectors.append(vector)
        
        print(f"向量提取完成，维度: {len(vectors[0])}")
        return np.array(vectors)
    
    def contrastive_learning(self, pos_vectors: np.ndarray, neg_vectors: np.ndarray, 
                           temperature: float = 0.07) -> Tuple[np.ndarray, np.ndarray]:
        """
        对比学习机制，增强价值观特征的表征区分度
        
        Args:
            pos_vectors: 正样本向量
            neg_vectors: 负样本向量
            temperature: 温度参数，控制对比学习的锐度
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 经过对比学习优化的正负样本向量
        """
        print("开始进行对比学习...")
        
        # 标准化向量
        pos_vectors = normalize(pos_vectors)
        neg_vectors = normalize(neg_vectors)
        
        # 计算正样本间的相似度矩阵
        pos_sim = cosine_similarity(pos_vectors)
        # 计算正负样本间的相似度矩阵
        pos_neg_sim = cosine_similarity(pos_vectors, neg_vectors)
        
        # 使用softmax调整相似度权重
        # 将正样本间相似度放大，负样本间相似度减小
        pos_sim_exp = np.exp(pos_sim / temperature)
        pos_neg_sim_exp = np.exp(-pos_neg_sim / temperature)
        
        # 对角线元素设为0，避免自己与自己的相似度影响
        np.fill_diagonal(pos_sim_exp, 0)
        
        # 根据相似度重新加权向量
        refined_pos_vectors = []
        for i in range(len(pos_vectors)):
            # 计算与所有正样本的加权平均（拉近正样本）
            pos_weights = pos_sim_exp[i] / (np.sum(pos_sim_exp[i]) + 1e-8)
            refined_pos = np.average(pos_vectors, axis=0, weights=pos_weights)
            
            # 减去与负样本的加权影响（推远负样本）
            neg_weights = np.mean(pos_neg_sim_exp[i]) / (len(neg_vectors) + 1e-8)
            avg_neg = np.mean(neg_vectors, axis=0)
            
            refined_vector = refined_pos - 0.1 * neg_weights * avg_neg
            refined_pos_vectors.append(refined_vector)
        
        refined_pos_vectors = np.array(refined_pos_vectors)
        
        # 同样处理负样本，使其远离正样本
        refined_neg_vectors = []
        for i in range(len(neg_vectors)):
            # 计算与所有负样本的加权平均（拉近负样本）
            neg_sim_exp = np.exp(cosine_similarity([neg_vectors[i]], neg_vectors)[0] / temperature)
            np.fill_diagonal(np.array([neg_sim_exp]), 0)
            neg_weights = neg_sim_exp / (np.sum(neg_sim_exp) + 1e-8)
            refined_neg = np.average(neg_vectors, axis=0, weights=neg_weights)
            
            # 减去与正样本的加权影响（推远正样本）
            pos_weights = np.mean(cosine_similarity([neg_vectors[i]], pos_vectors)[0]) / (len(pos_vectors) + 1e-8)
            avg_pos = np.mean(pos_vectors, axis=0)
            
            refined_vector = refined_neg - 0.1 * pos_weights * avg_pos
            refined_neg_vectors.append(refined_vector)
        
        refined_neg_vectors = np.array(refined_neg_vectors)
        
        print("对比学习完成")
        return refined_pos_vectors, refined_neg_vectors
    
    def cluster_vectors(self, vectors: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        对向量进行层次聚类分析
        
        Args:
            vectors: 待聚类的向量
            n_clusters: 聚类数量，如果为None则自动确定
            
        Returns:
            np.ndarray: 每个聚类的质心向量
        """
        print(f"开始对{len(vectors)}个向量进行聚类...")
        
        # 如果未指定聚类数量，使用启发式方法估算
        if n_clusters is None:
            n_clusters = max(3, len(vectors) // 10)  # 每10个样本大约1个聚类
        
        # 使用层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'  # Ward linkage minimizes within-cluster variance
        )
        
        cluster_labels = clustering.fit_predict(vectors)
        
        # 计算每个聚类的质心
        centroids = []
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_vectors = vectors[mask]
            if len(cluster_vectors) > 0:
                centroid = np.mean(cluster_vectors, axis=0)
                centroids.append(centroid)
        
        print(f"聚类完成，共{len(centroids)}个聚类中心")
        return np.array(centroids)
    
    def build_prototype_space(self, n_positive_clusters: int = 5, n_negative_clusters: int = 5) -> Dict:
        """
        构建价值观原型空间
        
        Args:
            n_positive_clusters: 正样本聚类数量
            n_negative_clusters: 负样本聚类数量
            
        Returns:
            Dict: 包含原型和反原型的字典
        """
        print("开始构建价值观原型空间...")
        
        # 提取正负样本文本
        positive_texts = self.data['question_1'].tolist()
        negative_texts = self.data['question_2'].tolist()
        
        # 提取正负样本向量
        print("提取正样本向量...")
        self.positive_vectors = self.extract_vectors(positive_texts)
        print("提取负样本向量...")
        self.negative_vectors = self.extract_vectors(negative_texts)
        
        # 对比学习优化
        self.positive_vectors, self.negative_vectors = self.contrastive_learning(
            self.positive_vectors, 
            self.negative_vectors
        )
        
        # 对正负样本向量进行聚类
        positive_prototypes = self.cluster_vectors(self.positive_vectors, n_positive_clusters)
        negative_prototypes = self.cluster_vectors(self.negative_vectors, n_negative_clusters)
        
        # 构建原型空间
        self.prototypes = {
            'positive_prototypes': positive_prototypes,
            'negative_prototypes': negative_prototypes,
            'positive_count': len(positive_prototypes),
            'negative_count': len(negative_prototypes)
        }
        
        print(f"原型空间构建完成:")
        print(f"- 正原型数量: {self.prototypes['positive_count']}")
        print(f"- 负原型数量: {self.prototypes['negative_count']}")
        
        return self.prototypes
    
    def save_prototypes(self, save_path: str):
        """
        保存原型向量到文件
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 保存正负原型
        np.save(os.path.join(save_path, 'positive_prototypes.npy'), self.prototypes['positive_prototypes'])
        np.save(os.path.join(save_path, 'negative_prototypes.npy'), self.prototypes['negative_prototypes'])
        
        print(f"原型已保存到: {save_path}")
    
    def load_prototypes(self, load_path: str):
        """
        从文件加载原型向量
        
        Args:
            load_path: 加载路径
        """
        positive_path = os.path.join(load_path, 'positive_prototypes.npy')
        negative_path = os.path.join(load_path, 'negative_prototypes.npy')
        
        if os.path.exists(positive_path) and os.path.exists(negative_path):
            self.prototypes['positive_prototypes'] = np.load(positive_path)
            self.prototypes['negative_prototypes'] = np.load(negative_path)
            self.prototypes['positive_count'] = len(self.prototypes['positive_prototypes'])
            self.prototypes['negative_count'] = len(self.prototypes['negative_prototypes'])
            
            print(f"原型已从 {load_path} 加载")
        else:
            print(f"找不到原型文件: {load_path}")


def get_prototype_vectors(csv_path: str, model_name: str = "t5-small", 
                         n_positive_clusters: int = 5, n_negative_clusters: int = 5,
                         save_path: str = None) -> Dict:
    """
    构建价值观原型向量库的主函数
    
    Args:
        csv_path: 包含正负样本的CSV文件路径
        model_name: 预训练模型名称
        n_positive_clusters: 正样本聚类数量
        n_negative_clusters: 负样本聚类数量
        save_path: 可选的保存路径
        
    Returns:
        Dict: 包含原型和反原型的字典
    """
    print("开始构建价值观原型向量库...")
    
    # 加载T5模型
    print(f"加载模型: {model_name}")
    model = T5Model(model_name)
    
    # 创建原型学习器
    learner = PrototypeLearner(model, csv_path)
    
    # 构建原型空间
    prototypes = learner.build_prototype_space(n_positive_clusters, n_negative_clusters)
    
    # 如果提供了保存路径，则保存原型
    if save_path:
        learner.save_prototypes(save_path)
    
    return prototypes


if __name__ == "__main__":
    # 示例：使用achievement_context_controlled.csv构建原型向量库
    csv_path = "d:/code/Python/value-alignment-prototype/data/controlled/achievement_context_controlled.csv"
    save_path = "d:/code/Python/value-alignment-prototype/prototypes"
    
    print("开始构建原型向量库...")
    prototypes = get_prototype_vectors(
        csv_path=csv_path,
        model_name="t5-small",
        n_positive_clusters=5,
        n_negative_clusters=5,
        save_path=save_path
    )
    
    print("原型向量库构建完成！")
    print(f"正原型数量: {prototypes['positive_count']}")
    print(f"负原型数量: {prototypes['negative_count']}")