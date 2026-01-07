import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict
import os
import warnings
import re
import json
warnings.filterwarnings('ignore')

# 修改导入方式，使用相对路径导入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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
        self.positive_cluster_labels = None
        self.negative_cluster_labels = None
        self.prototypes = {}
        
        # 从CSV路径中提取价值观名称
        self.value_name = self._extract_value_name(csv_path)
        print(f"检测到价值观名称: {self.value_name}")
        
        # 读取CSV数据
        self._load_data()
    
    def _extract_value_name(self, csv_path: str) -> str:
        """
        从CSV文件路径中提取价值观名称
        例如: achievement_context_controlled.csv -> achievement
        """
        filename = os.path.basename(csv_path)
        # 使用正则表达式提取价值观名称（第一个下划线前的部分）
        match = re.match(r'^([a-zA-Z]+)_', filename)
        if match:
            return match.group(1)
        else:
            # 如果无法从文件名中提取，使用默认名称
            return "default_value"
        
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
    
    def cluster_vectors(self, vectors: np.ndarray, n_clusters: int = None, texts: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        对向量进行层次聚类分析
        
        Args:
            vectors: 待聚类的向量
            n_clusters: 聚类数量，如果为None则自动确定
            texts: 对应的文本列表，用于后续分析
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 每个聚类的质心向量和聚类标签
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
        return np.array(centroids), cluster_labels
    
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
        
        # 对正负样本向量进行聚类，同时获取聚类标签
        positive_prototypes, self.positive_cluster_labels = self.cluster_vectors(
            self.positive_vectors, n_positive_clusters, positive_texts)
        negative_prototypes, self.negative_cluster_labels = self.cluster_vectors(
            self.negative_vectors, n_negative_clusters, negative_texts)
        
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
        保存原型向量到文件，使用价值观名称创建子目录
        """
        # 使用价值观名称创建子目录
        value_dir = os.path.join(save_path, self.value_name)
        os.makedirs(value_dir, exist_ok=True)
        
        # 保存正负原型
        positive_path = os.path.join(value_dir, f'{self.value_name}_positive_prototypes.npy')
        negative_path = os.path.join(value_dir, f'{self.value_name}_negative_prototypes.npy')
        np.save(positive_path, self.prototypes['positive_prototypes'])
        np.save(negative_path, self.prototypes['negative_prototypes'])
        
        print(f"原型已保存到: {value_dir}")
    
    def save_cluster_assignments(self, save_path: str):
        """
        保存聚类分配情况，包括每个聚类群对应的样本ID
        
        Args:
            save_path: 保存路径
        """
        if self.positive_cluster_labels is None or self.negative_cluster_labels is None:
            print("没有找到聚类标签数据，请先构建原型空间")
            return
        
        # 使用价值观名称创建子目录
        value_dir = os.path.join(save_path, self.value_name)
        os.makedirs(value_dir, exist_ok=True)
        
        # 获取数据ID列，如果不存在则使用索引
        if 'id' in self.data.columns:
            ids = self.data['id'].tolist()
        else:
            ids = list(range(len(self.data)))
        
        # 构建聚类分配字典
        cluster_assignments = {
            'value_type': self.value_name,
            'positive_clusters': {},
            'negative_clusters': {}
        }
        
        # 正样本聚类分配
        for idx, cluster_id in enumerate(self.positive_cluster_labels):
            cluster_key = f'cluster_{cluster_id}'
            if cluster_key not in cluster_assignments['positive_clusters']:
                cluster_assignments['positive_clusters'][cluster_key] = []
            cluster_assignments['positive_clusters'][cluster_key].append({
                'sample_id': ids[idx],
                'text': self.data.iloc[idx]['question_1'] if 'question_1' in self.data.columns else f"Sample {ids[idx]}"
            })
        
        # 负样本聚类分配
        for idx, cluster_id in enumerate(self.negative_cluster_labels):
            cluster_key = f'cluster_{cluster_id}'
            if cluster_key not in cluster_assignments['negative_clusters']:
                cluster_assignments['negative_clusters'][cluster_key] = []
            cluster_assignments['negative_clusters'][cluster_key].append({
                'sample_id': ids[idx],
                'text': self.data.iloc[idx]['question_2'] if 'question_2' in self.data.columns else f"Sample {ids[idx]}"
            })
        
        # 保存聚类分配情况
        cluster_path = os.path.join(value_dir, f'{self.value_name}_cluster_assignments.json')
        with open(cluster_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_assignments, f, ensure_ascii=False, indent=2)
        
        print(f"聚类分配情况已保存到: {cluster_path}")
        
        # 打印聚类统计信息
        print(f"\n聚类统计信息:")
        print(f"正样本聚类数: {len(cluster_assignments['positive_clusters'])}")
        for cluster_key, samples in cluster_assignments['positive_clusters'].items():
            print(f"  {cluster_key}: {len(samples)} 个样本")
        
        print(f"负样本聚类数: {len(cluster_assignments['negative_clusters'])}")
        for cluster_key, samples in cluster_assignments['negative_clusters'].items():
            print(f"  {cluster_key}: {len(samples)} 个样本")

    def load_prototypes(self, load_path: str):
        """
        从文件加载原型向量，使用价值观名称的子目录
        """
        # 使用价值观名称确定子目录
        value_dir = os.path.join(load_path, self.value_name)
        positive_path = os.path.join(value_dir, f'{self.value_name}_positive_prototypes.npy')
        negative_path = os.path.join(value_dir, f'{self.value_name}_negative_prototypes.npy')
        
        if os.path.exists(positive_path) and os.path.exists(negative_path):
            self.prototypes['positive_prototypes'] = np.load(positive_path)
            self.prototypes['negative_prototypes'] = np.load(negative_path)
            self.prototypes['positive_count'] = len(self.prototypes['positive_prototypes'])
            self.prototypes['negative_count'] = len(self.prototypes['negative_prototypes'])
            
            print(f"原型已从 {value_dir} 加载")
        else:
            print(f"找不到原型文件: {value_dir}")

    def visualize_prototypes(self, save_path: str = None):
        """
        可视化原型向量
        使用PCA将高维向量降到2D进行可视化，并使用价值观名称命名
        """
        if not self.prototypes or 'positive_prototypes' not in self.prototypes:
            print("没有找到原型数据，请先构建原型空间")
            return
        
        # 获取正负原型
        pos_prototypes = self.prototypes['positive_prototypes']
        neg_prototypes = self.prototypes['negative_prototypes']
        
        # 合并所有原型
        all_prototypes = np.vstack([pos_prototypes, neg_prototypes])
        
        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        prototypes_2d = pca.fit_transform(all_prototypes)
        
        # 分离正负原型的2D坐标
        pos_2d = prototypes_2d[:len(pos_prototypes)]
        neg_2d = prototypes_2d[len(pos_prototypes):]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制正原型（用蓝色圆点表示）
        plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c='blue', label='Positive Prototypes', alpha=0.7, s=100)
        
        # 绘制负原型（用红色叉号表示）
        plt.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', label='Negative Prototypes', alpha=0.7, marker='x', s=100)
        
        # 添加标题和标签
        plt.title(f'Visualization of {self.value_name.capitalize()} Value Alignment Prototypes\n(PCA 2D Projection)', fontsize=14)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 如果提供了保存路径，则使用价值观名称创建子目录并命名文件
        if save_path:
            # 使用价值观名称创建子目录
            value_dir = os.path.dirname(save_path)
            value_subdir = os.path.join(value_dir, self.value_name)
            os.makedirs(value_subdir, exist_ok=True)
            
            # 生成使用价值观名称的可视化图
            vis_path = os.path.join(value_subdir, f'{self.value_name}_prototype_visualization.png')
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            print(f"原型可视化图已保存到: {vis_path}")
        else:
            plt.show()
        
        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.2f}")
    
    def evaluate_prototype_quality(self):
        """
        评估原型构建的质量
        评估指标包括：
        1. 正负原型之间的分离度
        2. 正负原型内部的聚类质量
        """
        if not self.prototypes or 'positive_prototypes' not in self.prototypes:
            print("没有找到原型数据，请先构建原型空间")
            return None
        
        pos_prototypes = self.prototypes['positive_prototypes']
        neg_prototypes = self.prototypes['negative_prototypes']
        
        print("正在评估原型构建质量...")
        
        # 1. 计算正负原型之间的分离度
        # 计算正原型和负原型的中心点
        pos_center = np.mean(pos_prototypes, axis=0)
        neg_center = np.mean(neg_prototypes, axis=0)
        
        # 计算正负原型中心之间的欧氏距离
        center_distance = np.linalg.norm(pos_center - neg_center)
        
        # 计算正负原型中心之间的余弦相似度
        cos_sim = cosine_similarity([pos_center], [neg_center])[0][0]
        
        # 2. 计算正负原型内部的聚类质量
        # 计算正原型到其中心的平均距离
        pos_distances = [np.linalg.norm(proto - pos_center) for proto in pos_prototypes]
        avg_pos_intra_distance = np.mean(pos_distances)
        
        # 计算负原型到其中心的平均距离
        neg_distances = [np.linalg.norm(proto - neg_center) for proto in neg_prototypes]
        avg_neg_intra_distance = np.mean(neg_distances)
        
        # 3. 计算分离度与内部凝聚度的比率（越大越好）
        if avg_pos_intra_distance == 0 or avg_neg_intra_distance == 0:
            separation_to_cohesion_ratio = float('inf')
        else:
            avg_intra_distance = (avg_pos_intra_distance + avg_neg_intra_distance) / 2
            separation_to_cohesion_ratio = center_distance / avg_intra_distance
        
        # 4. 计算整体聚类的轮廓系数
        from sklearn.metrics import silhouette_score
        all_prototypes = np.vstack([pos_prototypes, neg_prototypes])
        labels = [0] * len(pos_prototypes) + [1] * len(neg_prototypes)
        silhouette_avg = silhouette_score(all_prototypes, labels)
        
        quality_metrics = {
            'center_distance': center_distance,  # 中心距离
            'cosine_similarity': cos_sim,  # 中心余弦相似度
            'avg_pos_intra_distance': avg_pos_intra_distance,  # 正原型内部平均距离
            'avg_neg_intra_distance': avg_neg_intra_distance,  # 负原型内部平均距离
            'separation_to_cohesion_ratio': separation_to_cohesion_ratio,  # 分离度与凝聚度比率
            'silhouette_score': silhouette_avg  # 轮廓系数
        }
        
        print("\n原型构建质量评估结果:")
        print(f"正负原型中心距离: {center_distance:.4f}")
        print(f"正负原型中心余弦相似度: {cos_sim:.4f}")
        print(f"正原型内部平均距离: {avg_pos_intra_distance:.4f}")
        print(f"负原型内部平均距离: {avg_neg_intra_distance:.4f}")
        print(f"分离度与凝聚度比率: {separation_to_cohesion_ratio:.4f}")
        print(f"轮廓系数: {silhouette_avg:.4f}")
        
        print("\n质量评估说明:")
        print("- 中心距离越大，表示正负原型分离越明显")
        print("- 余弦相似度越小(越负)，表示正负原型差异越大")
        print("- 内部平均距离越小，表示原型内部越凝聚")
        print("- 分离度与凝聚度比率越大，表示原型质量越高")
        print("- 轮廓系数越接近1，表示聚类效果越好")
        
        # 计算标准化的原型质量得分
        normalized_score = self._calculate_normalized_quality_score(quality_metrics)
        print(f"\n原型质量最终指标: {normalized_score:.4f}")
        
        return quality_metrics, normalized_score

    def _calculate_normalized_quality_score(self, metrics: Dict) -> float:
        """
        计算标准化的原型质量得分，将每个指标转换为0-1得分并加权
        
        Args:
            metrics: 包含质量评估指标的字典
            
        Returns:
            float: 0-1范围内的最终质量得分
        """
        # 指标权重设置
        weights = {
            'center_distance': 0.25,    # 中心距离权重
            'cosine_similarity': 0.25,  # 余弦相似度权重（负相关，需取反）
            'avg_pos_intra_distance': 0.15,  # 正原型内部距离权重（负相关）
            'avg_neg_intra_distance': 0.15,  # 负原型内部距离权重（负相关）
            'silhouette_score': 0.20    # 轮廓系数权重
        }
        
        # 归一化参数（基于典型值范围）
        max_center_distance = 5.0  # 假设最大中心距离
        max_intra_distance = 2.0   # 假设最大内部距离
        max_silhouette = 1.0       # 轮廓系数最大值
        
        # 计算各指标的0-1标准化得分
        # 中心距离：越大越好，直接归一化
        center_distance_score = min(metrics['center_distance'] / max_center_distance, 1.0)
        
        # 余弦相似度：越小越好（负值更佳），转换为正值后取反归一化
        # cos_sim越负越好，所以计算为 (1 - cos_sim)/2，将其映射到0-1范围
        cos_sim_score = (1 - metrics['cosine_similarity']) / 2.0
        cos_sim_score = max(0.0, min(cos_sim_score, 1.0))  # 确保在0-1范围内
        
        # 正原型内部距离：越小越好，取倒数或线性转换
        avg_pos_intra_score = max(0.0, min(1.0 - metrics['avg_pos_intra_distance'] / max_intra_distance, 1.0))
        
        # 负原型内部距离：越小越好，取倒数或线性转换
        avg_neg_intra_score = max(0.0, min(1.0 - metrics['avg_neg_intra_distance'] / max_intra_distance, 1.0))
        
        # 轮廓系数：越大越好，取绝对值并归一化
        silhouette_score = (metrics['silhouette_score'] + 1) / 2  # 将[-1,1]映射到[0,1]
        
        # 计算加权总分
        # 注意：内部距离只计算一次，所以平均它们的得分
        total_score = (
            weights['center_distance'] * center_distance_score +
            weights['cosine_similarity'] * cos_sim_score +
            (weights['avg_pos_intra_distance'] + weights['avg_neg_intra_distance']) / 2 * 
            (avg_pos_intra_score + avg_neg_intra_score) / 2 +
            weights['silhouette_score'] * silhouette_score
        )
        
        return total_score


def get_prototype_vectors(csv_path: str, model_name: str = "t5-3b", 
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
        # 同时保存聚类分配情况
        learner.save_cluster_assignments(save_path)
    
    return prototypes


if __name__ == "__main__":
    # 示例：使用achievement_context_controlled.csv构建原型向量库
    # 构建相对于当前脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "..", "data", "controlled", "achievement_context_controlled.csv")
    save_path = os.path.join(script_dir, "prototypes")
    
    print("开始构建原型向量库...")
    
    # 加载T5模型
    print(f"加载模型: t5-3b")
    model = T5Model("t5-3b")
    
    # 创建原型学习器并构建原型空间
    learner = PrototypeLearner(model, csv_path)
    prototypes = learner.build_prototype_space(5, 5)  # 5个正样本聚类，5个负样本聚类
    
    # 保存原型
    learner.save_prototypes(save_path)
    
    # 保存聚类分配情况
    learner.save_cluster_assignments(save_path)
    
    print("原型向量库构建完成！")
    print(f"正原型数量: {prototypes['positive_count']}")
    print(f"负原型数量: {prototypes['negative_count']}")
    
    # 评估原型质量
    print("\n开始评估原型构建质量...")
    quality_metrics = learner.evaluate_prototype_quality()
    
    # 可视化原型
    print("开始可视化原型...")
    
    # 生成可视化图（使用价值观子目录保存）
    learner.visualize_prototypes(save_path=os.path.join(script_dir, "prototypes", "visualization.png"))
