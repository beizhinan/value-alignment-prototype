import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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

# 设置中文字体支持 - 修正版
try:
    # 尝试使用系统中可用的中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print("中文字体设置成功")
except:
    print("中文字体设置失败，使用默认字体")

# 修改导入方式，使用相对路径导入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.model import T5Model


class PrototypeLearner:
    """
    原型学习类，用于构建价值观原型向量库
    """
    
    def __init__(self, model: T5Model, csv_path: str, batch_size: int = 32):
        """
        初始化原型学习器
        
        Args:
            model: 已加载的T5模型
            csv_path: 包含正负样本的CSV文件路径
            batch_size: 批处理大小（默认32）
        """
        self.model = model
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.data = None
        self.positive_vectors = None
        self.negative_vectors = None
        self.positive_cluster_labels = None
        self.negative_cluster_labels = None
        self.prototypes = {}
        
        # 从CSV路径中提取价值观名称
        self.value_name = self._extract_value_name(csv_path)
        print(f"检测到价值观名称: {self.value_name}")
        print(f"批处理大小: {batch_size}")
        
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
        使用T5模型提取文本向量（单条处理，兼容旧代码）
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 文本向量矩阵
        """
        print(f"注意：使用单条处理模式，速度较慢，建议使用extract_vectors_batch方法")
        return self.extract_vectors_batch(texts, batch_size=1)
    
    def extract_vectors_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        批量提取文本向量，大幅提高处理速度
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小，如果为None则使用实例的batch_size
            
        Returns:
            np.ndarray: 文本向量矩阵
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        print(f"正在批量提取{len(texts)}个文本的向量表示，批大小: {batch_size}...")
        
        all_vectors = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            if batch_num % 5 == 0 or batch_num == total_batches:
                print(f"处理进度: {batch_num}/{total_batches} 批次 ({len(batch_texts)}条文本)")
            
            # 批量处理：对每个文本单独获取隐藏状态并池化
            batch_vectors = []
            for text in batch_texts:
                # 获取编码器最后一层的隐藏状态
                last_hidden_state = self.model.get_encoder_last_hidden_state(text)
                
                # 对序列维度进行平均池化，得到固定长度的向量
                pooled_vector = torch.mean(last_hidden_state, dim=1).squeeze(0)
                
                # 转换为numpy数组
                vector = pooled_vector.detach().cpu().numpy()
                batch_vectors.append(vector)
            
            all_vectors.extend(batch_vectors)
        
        print(f"批量向量提取完成，总维度: {len(all_vectors)} × {len(all_vectors[0])}")
        return np.array(all_vectors)
    
    def smooth_contrastive_learning(self, pos_vectors: np.ndarray, neg_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        平滑对比学习机制，避免温度调节，适用于重叠度高的价值观数据
        
        Args:
            pos_vectors: 正样本向量
            neg_vectors: 负样本向量
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 经过对比学习优化的正负样本向量
        """
        print("开始进行平滑对比学习...")
        
        # 标准化向量
        pos_vectors_norm = normalize(pos_vectors)
        neg_vectors_norm = normalize(neg_vectors)
        
        # 计算全局中心
        pos_center = np.mean(pos_vectors_norm, axis=0)
        neg_center = np.mean(neg_vectors_norm, axis=0)
        
        # 平滑更新：所有样本向类中心收缩，同时远离对方类中心
        refined_pos = []
        for vec in pos_vectors_norm:
            # 向正样本中心靠拢，远离负样本中心
            # 使用线性组合，避免极端值
            new_vec = 0.8 * vec + 0.1 * pos_center - 0.1 * neg_center
            refined_pos.append(new_vec)
        
        refined_neg = []
        for vec in neg_vectors_norm:
            # 向负样本中心靠拢，远离正样本中心
            new_vec = 0.8 * vec + 0.1 * neg_center - 0.1 * pos_center
            refined_neg.append(new_vec)
        
        # 重新标准化
        refined_pos = normalize(np.array(refined_pos))
        refined_neg = normalize(np.array(refined_neg))
        
        print("平滑对比学习完成")
        return refined_pos, refined_neg
    
    def contrastive_learning(self, pos_vectors: np.ndarray, neg_vectors: np.ndarray, 
                           temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        对比学习机制（兼容性版本，默认使用平滑对比）
        
        Args:
            pos_vectors: 正样本向量
            neg_vectors: 负样本向量
            temperature: 温度参数，当temperature=1时使用平滑对比
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 经过对比学习优化的正负样本向量
        """
        if temperature == 1.0:
            # 使用平滑对比学习
            return self.smooth_contrastive_learning(pos_vectors, neg_vectors)
        else:
            print(f"使用标准对比学习，温度系数: {temperature}")
            
            # 标准化向量
            pos_vectors = normalize(pos_vectors)
            neg_vectors = normalize(neg_vectors)
            
            # 计算正样本间的相似度矩阵
            pos_sim = cosine_similarity(pos_vectors)
            # 计算正负样本间的相似度矩阵
            pos_neg_sim = cosine_similarity(pos_vectors, neg_vectors)
            
            # 使用softmax调整相似度权重
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
            linkage='ward'  
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
    
    def extract_all_vectors(self):
        """
        提取所有正负样本向量（如果尚未提取）
        这个方法确保向量只提取一次，使用批量处理
        """
        if self.positive_vectors is None or self.negative_vectors is None:
            print("批量提取所有样本向量...")
            positive_texts = self.data['question_1'].tolist()
            negative_texts = self.data['question_2'].tolist()
            
            print(f"提取{len(positive_texts)}个正样本向量...")
            self.positive_vectors = self.extract_vectors_batch(positive_texts)
            print(f"提取{len(negative_texts)}个负样本向量...")
            self.negative_vectors = self.extract_vectors_batch(negative_texts)
        else:
            print("向量已经提取过，直接使用缓存")
    
    def build_prototype_space(self, n_positive_clusters: int = 10, n_negative_clusters: int = 10, temperature: float = 1.0) -> Dict:
        """
        构建价值观原型空间
        
        Args:
            n_positive_clusters: 正样本聚类数量（默认10）
            n_negative_clusters: 负样本聚类数量（默认10）
            temperature: 对比学习温度系数（默认1.0，使用平滑对比）
            
        Returns:
            Dict: 包含原型和反原型的字典
        """
        print(f"开始构建价值观原型空间...")
        print(f"参数: 正样本聚类数={n_positive_clusters}, 负样本聚类数={n_negative_clusters}, 温度系数={temperature}")
        
        # 提取所有样本向量（确保只提取一次）
        self.extract_all_vectors()
        
        # 对比学习优化（默认使用平滑对比）
        processed_pos_vectors, processed_neg_vectors = self.contrastive_learning(
            self.positive_vectors.copy(), 
            self.negative_vectors.copy(),
            temperature
        )
        
        # 对正负样本向量进行聚类，同时获取聚类标签
        positive_prototypes, self.positive_cluster_labels = self.cluster_vectors(
            processed_pos_vectors, n_positive_clusters, self.data['question_1'].tolist())
        negative_prototypes, self.negative_cluster_labels = self.cluster_vectors(
            processed_neg_vectors, n_negative_clusters, self.data['question_2'].tolist())
        
        # 构建原型空间
        self.prototypes = {
            'positive_prototypes': positive_prototypes,
            'negative_prototypes': negative_prototypes,
            'positive_count': len(positive_prototypes),
            'negative_count': len(negative_prototypes),
            'config': {
                'temperature': temperature,
                'n_positive_clusters': n_positive_clusters,
                'n_negative_clusters': n_negative_clusters,
                'batch_size': self.batch_size
            }
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
        
        # 保存配置
        if 'config' in self.prototypes:
            config_path = os.path.join(value_dir, f'{self.value_name}_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.prototypes['config'], f, ensure_ascii=False, indent=2)
        
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
        显示所有样本点并用颜色区分不同的聚类
        """
        if not self.prototypes or 'positive_prototypes' not in self.prototypes:
            print("没有找到原型数据，请先构建原型空间")
            return
        
        # 获取正负原型
        pos_prototypes = self.prototypes['positive_prototypes']
        neg_prototypes = self.prototypes['negative_prototypes']
        
        # 使用PCA降维到2D
        all_prototypes = np.vstack([pos_prototypes, neg_prototypes])
        pca = PCA(n_components=2)
        prototypes_2d = pca.fit_transform(all_prototypes)
        
        # 分离正负原型的2D坐标
        pos_2d = prototypes_2d[:len(pos_prototypes)]
        neg_2d = prototypes_2d[len(pos_prototypes):]
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制正原型（用蓝色圆点表示）
        plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c='blue', label='Positive Prototypes', alpha=0.7, s=100)
        
        # 绘制负原型（用红色叉号表示）
        plt.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', label='Negative Prototypes', alpha=0.7, marker='x', s=100)
        
        # 如果存在聚类标签，也显示聚类结果
        if hasattr(self, 'positive_cluster_labels') and hasattr(self, 'negative_cluster_labels'):
            # 获取原始向量并降维以可视化聚类
            if self.positive_vectors is not None and self.negative_vectors is not None:
                # 对原始向量进行PCA降维
                all_original_vectors = np.vstack([self.positive_vectors, self.negative_vectors])
                original_vectors_2d = pca.transform(all_original_vectors)
                
                # 分离正负样本的2D坐标
                pos_original_2d = original_vectors_2d[:len(self.positive_vectors)]
                neg_original_2d = original_vectors_2d[len(self.positive_vectors):]
                
                # 使用不同颜色绘制各个聚类
                unique_pos_clusters = np.unique(self.positive_cluster_labels)
                unique_neg_clusters = np.unique(self.negative_cluster_labels)
                
                # 为正样本聚类分配颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_pos_clusters) + len(unique_neg_clusters)))
                
                # 绘制正样本聚类
                for i, cluster_id in enumerate(unique_pos_clusters):
                    mask = self.positive_cluster_labels == cluster_id
                    cluster_points = pos_original_2d[mask]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              c=[colors[i]], label=f'Pos Cluster {cluster_id}', alpha=0.6, s=30)
                
                # 绘制负样本聚类
                for i, cluster_id in enumerate(unique_neg_clusters):
                    mask = self.negative_cluster_labels == cluster_id
                    cluster_points = neg_original_2d[mask]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              c=[colors[len(unique_pos_clusters) + i]], label=f'Neg Cluster {cluster_id}', alpha=0.6, s=30)
        
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
            plt.close()
        else:
            plt.tight_layout()
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
        
        # 3. 计算整体聚类的轮廓系数
        from sklearn.metrics import silhouette_score
        all_prototypes = np.vstack([pos_prototypes, neg_prototypes])
        labels = [0] * len(pos_prototypes) + [1] * len(neg_prototypes)
        silhouette_avg = silhouette_score(all_prototypes, labels)
        
        quality_metrics = {
            'center_distance': center_distance,  # 中心距离
            'cosine_similarity': cos_sim,  # 中心余弦相似度
            'avg_pos_intra_distance': avg_pos_intra_distance,  # 正原型内部平均距离
            'avg_neg_intra_distance': avg_neg_intra_distance,  # 负原型内部平均距离
            'silhouette_score': silhouette_avg  # 轮廓系数
        }
        
        # 计算标准化的原型质量得分
        normalized_score = self._calculate_normalized_quality_score(quality_metrics)
        print(f"\n原型质量最终指标: {normalized_score:.4f}")
        
        return quality_metrics, normalized_score

    def _calculate_normalized_quality_score(self, metrics: Dict) -> float:
        """
        使用信息重叠更少的指标进行原型质量评估
        移除分离度指标，仅使用凝聚度和平衡性两个指标
        
        Args:
            metrics: 包含质量指标的字典
            
        Returns:
            float: 归一化后的质量得分 [0, 1]
        """
        def sigmoid_norm(x, midpoint, steepness=10):
            """使用sigmoid函数进行平滑归一化"""
            return 1 / (1 + np.exp(-steepness * (x - midpoint)))
        
        # 移除分离度指标，仅使用凝聚度和平衡性指标
        # 调整权重分配，使凝聚度指标占主导地位，但仍保持平衡性指标的重要性
        weights = {
            'cohesion_score': 0.5,    # 凝聚度综合指标
            'balance_score': 0.5      # 平衡性指标
        }
        
        # 1. 凝聚度指标（考虑正负差异）
        avg_intra = (metrics['avg_pos_intra_distance'] + 
                     metrics['avg_neg_intra_distance']) / 2
                     
        # 动态计算midpoint值，使用平均内部距离的某个百分位数作为参考
        # 这里使用0.2作为默认值，但可以根据经验或数据分布调整
        midpoint = avg_intra * 0.8  # 使用实际观测值的80%作为"好坏"分界点
        
        cohesion_score = 1 - sigmoid_norm(avg_intra, midpoint=max(0.05, midpoint))  # 确保midpoint不会太小
        
        # 2. 平衡性指标（正负原型数量和质量是否均衡）
        pos_intra = metrics['avg_pos_intra_distance']
        neg_intra = metrics['avg_neg_intra_distance']
        intra_sum = pos_intra + neg_intra + 1e-8
        balance_score = 1 - abs(pos_intra - neg_intra) / intra_sum
        
        # 计算总分
        total_score = (
            weights['cohesion_score'] * cohesion_score +
            weights['balance_score'] * balance_score
        )
        
        # 确保分数在[0, 1]范围内
        return max(0.0, min(1.0, total_score))

    def _sigmoid_norm(self, x, midpoint, steepness=10):
        """使用sigmoid函数进行平滑归一化"""
        return 1 / (1 + np.exp(-steepness * (x - midpoint)))

    def compare_prototype_quality(self, temperature_values: List[float], cluster_values: List[int], save_path: str = None):
        """
        对比不同温度系数和聚类簇数下的指标情况，并作图可视化
        修改：使用已提取的基础向量，避免重复提取
        
        Args:
            temperature_values: 温度系数列表
            cluster_values: 聚类簇数列表
            save_path: 保存路径
            
        Returns:
            tuple: (最优参数, 最优指标, 所有结果)
        """
        print(f"开始对比不同温度系数 {temperature_values} 和聚类簇数 {cluster_values} 下的指标情况...")
        print(f"总共需要测试 {len(temperature_values) * len(cluster_values)} 种参数组合")
        
        # 确保向量已经提取（只提取一次）
        self.extract_all_vectors()
        
        # 存储所有参数组合的结果
        results = []
        best_score = -1
        best_params = None
        best_prototypes = None
        
        # 遍历所有参数组合
        for temp in temperature_values:
            for n_clusters in cluster_values:
                print(f"\n测试参数: 温度系数={temp}, 聚类数={n_clusters}(正负样本)")
                
                # 使用缓存的基础向量进行对比学习（使用copy避免修改原始向量）
                processed_pos_vectors, processed_neg_vectors = self.contrastive_learning(
                    self.positive_vectors.copy(), 
                    self.negative_vectors.copy(),
                    temp
                )
                
                # 对正负样本向量进行聚类
                positive_prototypes, _ = self.cluster_vectors(
                    processed_pos_vectors, n_clusters, None)  # 不再传递文本列表
                negative_prototypes, _ = self.cluster_vectors(
                    processed_neg_vectors, n_clusters, None)  # 不再传递文本列表
                
                # 临时存储原型
                temp_prototypes = {
                    'positive_prototypes': positive_prototypes,
                    'negative_prototypes': negative_prototypes,
                    'positive_count': len(positive_prototypes),
                    'negative_count': len(negative_prototypes)
                }
                
                # 评估原型质量
                original_prototypes = self.prototypes
                self.prototypes = temp_prototypes
                quality_metrics, score = self.evaluate_prototype_quality()
                
                # 记录结果
                result = {
                    'temperature': temp,
                    'n_clusters': n_clusters,
                    'quality_metrics': quality_metrics,
                    'score': score
                }
                results.append(result)
                
                print(f"当前参数质量得分: {score:.4f}")
                
                # 更新最优参数
                if score > best_score:
                    best_score = score
                    best_params = {'temperature': temp, 'n_clusters': n_clusters}
                    # 保存最优原型
                    best_prototypes = temp_prototypes.copy()
                
                # 恢复原始原型
                self.prototypes = original_prototypes
        
        print(f"\n所有参数组合测试完成！")
        print(f"最优参数: 温度系数={best_params['temperature']}, 聚类数={best_params['n_clusters']}, 得分={best_score:.4f}")
        
        # 保存最优原型
        if save_path and best_prototypes:
            # 临时更新原型为最优原型
            original_prototypes = self.prototypes
            self.prototypes = best_prototypes
            
            self.save_prototypes(save_path)
            
            # 注意：这里不再调用save_cluster_assignments，因为没有计算聚类标签
            # 但可以在构建最终原型时再计算
            
            # 恢复原始原型
            self.prototypes = original_prototypes
            
            print(f"最优原型已保存到: {save_path}")
        
        # 可视化结果
        self._visualize_comparison_results(results, temperature_values, cluster_values, save_path)
        
        return best_params, best_score, results

    def _visualize_comparison_results(self, results: List[Dict], temperature_values: List[float], cluster_values: List[int], save_path: str = None):
        """
        可视化对比结果 - 当温度系数只有1.0时使用条形图，否则使用热力图
        """
        print("正在生成对比结果可视化图...")
        
        # 检查是否只有一个温度值且为1.0
        temp_unique = sorted(set([r['temperature'] for r in results]))
        cluster_unique = sorted(set([r['n_clusters'] for r in results]))
        
        if len(temp_unique) == 1 and temp_unique[0] == 1.0:
            # 只有一个温度值1.0，使用条形图
            self._visualize_with_bar_charts(results, cluster_unique, save_path)
        else:
            # 多个温度值，使用热力图
            self._visualize_with_heatmaps(results, temp_unique, cluster_unique, save_path)

    def _visualize_with_bar_charts(self, results: List[Dict], cluster_values: List[int], save_path: str = None):
        """
        使用条形图可视化单温度下的结果
        """
        print("使用条形图可视化单温度下的结果...")
        
        # 准备数据
        cluster_values_sorted = sorted(cluster_values)
        scores = []
        cohesion_scores = []
        balance_scores = []
        
        for n_clusters in cluster_values_sorted:
            result = next((r for r in results if r['n_clusters'] == n_clusters and r['temperature'] == 1.0), None)
            if result:
                scores.append(result['score'])
                
                # 提取指标并计算新指标的组成部分
                metrics = result['quality_metrics']
                
                # 计算凝聚度综合得分（平均内部距离的倒数，值越小越好）
                avg_intra = (metrics['avg_pos_intra_distance'] + metrics['avg_neg_intra_distance']) / 2
                
                # 使用固定的sigmoid函数，而不是调用实例方法
                def sigmoid_norm(x, midpoint, steepness=10):
                    """使用sigmoid函数进行平滑归一化"""
                    return 1 / (1 + np.exp(-steepness * (x - midpoint)))
                
                # 动态计算midpoint值
                midpoint = avg_intra * 0.8
                cohesion_scores.append(1 - sigmoid_norm(avg_intra, max(0.05, midpoint)) if avg_intra < 1 else 0)
                
                # 计算平衡性得分
                pos_intra = metrics['avg_pos_intra_distance']
                neg_intra = metrics['avg_neg_intra_distance']
                intra_sum = pos_intra + neg_intra + 1e-8
                balance = 1 - abs(pos_intra - neg_intra) / intra_sum
                balance_scores.append(balance)
            else:
                scores.append(0)
                cohesion_scores.append(0)
                balance_scores.append(0)
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{self.value_name.capitalize()} - Parameter Comparison Bar Charts (Temperature=1.0)', 
                     fontsize=16, fontweight='bold')
        
        # 总体得分条形图
        bars1 = axes[0].bar(cluster_values_sorted, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0].set_title('Overall Quality Score', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Clusters', fontsize=11)
        axes[0].set_ylabel('Score', fontsize=11)
        axes[0].set_xticks(cluster_values_sorted)  # 设置横坐标刻度为聚类数的实际值
        axes[0].grid(axis='y', alpha=0.3)
        # 在每个条形上添加数值
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                            f'{score:.3f}',
                            ha="center", va="bottom", fontsize=9, fontweight='bold')
        
        # 凝聚度得分条形图
        bars2 = axes[1].bar(cluster_values_sorted, cohesion_scores, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        axes[1].set_title('Cohesion Score (1 - Avg Intra Distance)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Clusters', fontsize=11)
        axes[1].set_ylabel('Score', fontsize=11)
        axes[1].set_xticks(cluster_values_sorted)  # 设置横坐标刻度为聚类数的实际值
        axes[1].grid(axis='y', alpha=0.3)
        # 在每个条形上添加数值
        for bar, score in zip(bars2, cohesion_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{score:.3f}',
                            ha="center", va="bottom", fontsize=9, fontweight='bold')
        
        # 平衡性得分条形图
        bars3 = axes[2].bar(cluster_values_sorted, balance_scores, color='coral', edgecolor='darkred', alpha=0.7)
        axes[2].set_title('Balance Score (1 - |Pos-Neg|/Sum)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Clusters', fontsize=11)
        axes[2].set_ylabel('Score', fontsize=11)
        axes[2].set_xticks(cluster_values_sorted)  # 设置横坐标刻度为聚类数的实际值
        axes[2].grid(axis='y', alpha=0.3)
        # 在每个条形上添加数值
        for bar, score in zip(bars3, balance_scores):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                            f'{score:.3f}',
                            ha="center", va="bottom", fontsize=9, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            value_dir = os.path.join(save_path, self.value_name)
            os.makedirs(value_dir, exist_ok=True)
            plot_path = os.path.join(value_dir, f'{self.value_name}_parameter_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"参数对比可视化图已保存到: {plot_path}")
            plt.close()
        else:
            plt.show()

    def _visualize_with_heatmaps(self, results: List[Dict], temp_unique: List[float], cluster_unique: List[int], save_path: str = None):
        """
        使用热力图可视化多温度下的结果
        适配新的评估指标
        """
        print("使用热力图可视化多温度下的结果...")
        
        # 准备数据
        temp_mesh, cluster_mesh = np.meshgrid(temp_unique, cluster_unique)
        score_mesh = np.zeros_like(temp_mesh, dtype=float)
        # 移除分离度得分热力图，仅保留凝聚度和平衡性得分
        cohesion_mesh = np.zeros_like(temp_mesh, dtype=float)
        balance_mesh = np.zeros_like(temp_mesh, dtype=float)
        
        # 填充网格数据
        for i, n_clusters in enumerate(cluster_unique):
            for j, temp in enumerate(temp_unique):
                result = next((r for r in results if r['temperature'] == temp and r['n_clusters'] == n_clusters), None)
                if result:
                    score_mesh[i, j] = result['score']
                    
                    # 提取指标并计算新指标的组成部分
                    metrics = result['quality_metrics']
                    
                    # 凝聚度得分：1 - avg_intra_distance（归一化）
                    avg_intra = (metrics['avg_pos_intra_distance'] + metrics['avg_neg_intra_distance']) / 2
                    cohesion_mesh[i, j] = 1 - avg_intra if avg_intra < 1 else 0
                    
                    # 平衡性得分
                    pos_intra = metrics['avg_pos_intra_distance']
                    neg_intra = metrics['avg_neg_intra_distance']
                    intra_sum = pos_intra + neg_intra + 1e-8
                    balance_mesh[i, j] = 1 - abs(pos_intra - neg_intra) / intra_sum
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # 使用英文标题避免乱码
        fig.suptitle(f'{self.value_name.capitalize()} - Parameter Comparison Heatmaps', fontsize=16, fontweight='bold')
        
        # 总体得分热力图
        im1 = axes[0].imshow(score_mesh, cmap='viridis', aspect='auto', origin='lower')
        axes[0].set_xticks(range(len(temp_unique)))
        axes[0].set_xticklabels([f"{t}" for t in temp_unique])
        axes[0].set_yticks(range(len(cluster_unique)))
        axes[0].set_yticklabels([str(c) for c in cluster_unique])
        axes[0].set_title('Overall Quality Score', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Temperature', fontsize=11)
        axes[0].set_ylabel('Clusters', fontsize=11)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.ax.set_ylabel('Score', rotation=270, labelpad=15)
        # 在每个格子中添加数值
        for i in range(len(cluster_unique)):
            for j in range(len(temp_unique)):
                axes[0].text(j, i, f"{score_mesh[i, j]:.3f}",
                                ha="center", va="center", color="white", 
                                fontsize=8, fontweight='bold')
        
        # 凝聚度得分热力图
        im2 = axes[1].imshow(cohesion_mesh, cmap='RdYlBu_r', aspect='auto', origin='lower')
        axes[1].set_xticks(range(len(temp_unique)))
        axes[1].set_xticklabels([f"{t}" for t in temp_unique])
        axes[1].set_yticks(range(len(cluster_unique)))
        axes[1].set_yticklabels([str(c) for c in cluster_unique])
        axes[1].set_title('Cohesion Score (1 - Avg Intra Dist)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Temperature', fontsize=11)
        axes[1].set_ylabel('Clusters', fontsize=11)
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.ax.set_ylabel('Score', rotation=270, labelpad=15)
        # 在每个格子中添加数值
        for i in range(len(cluster_unique)):
            for j in range(len(temp_unique)):
                text_color = 'white' if cohesion_mesh[i, j] > 0.5 else 'black'
                axes[1].text(j, i, f"{cohesion_mesh[i, j]:.3f}",
                                ha="center", va="center", color=text_color, 
                                fontsize=8, fontweight='bold')
        
        # 平衡性得分热力图
        im3 = axes[2].imshow(balance_mesh, cmap='viridis', aspect='auto', origin='lower')
        axes[2].set_xticks(range(len(temp_unique)))
        axes[2].set_xticklabels([f"{t}" for t in temp_unique])
        axes[2].set_yticks(range(len(cluster_unique)))
        axes[2].set_yticklabels([str(c) for c in cluster_unique])
        axes[2].set_title('Balance Score (1 - |Pos-Neg|/Sum)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Temperature', fontsize=11)
        axes[2].set_ylabel('Clusters', fontsize=11)
        cbar3 = plt.colorbar(im3, ax=axes[2])
        cbar3.ax.set_ylabel('Score', rotation=270, labelpad=15)
        # 在每个格子中添加数值
        for i in range(len(cluster_unique)):
            for j in range(len(temp_unique)):
                axes[2].text(j, i, f"{balance_mesh[i, j]:.3f}",
                                ha="center", va="center", color="white", 
                                fontsize=8, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            value_dir = os.path.join(save_path, self.value_name)
            os.makedirs(value_dir, exist_ok=True)
            plot_path = os.path.join(value_dir, f'{self.value_name}_parameter_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"参数对比可视化图已保存到: {plot_path}")
            plt.close()
        else:
            plt.show()
    
    def analyze_data_characteristics(self):
        """
        分析数据特征，为选择平滑对比学习提供依据
        修改：使用已提取的向量，避免重复提取
        """
        print(f"\n分析数据特征: {self.value_name}")
        
        # 确保向量已经提取（只提取一次）
        self.extract_all_vectors()
        
        # 标准化向量
        pos_vectors_norm = normalize(self.positive_vectors)
        neg_vectors_norm = normalize(self.negative_vectors)
        
        # 计算正负样本间的相似度
        pos_neg_sim = cosine_similarity(pos_vectors_norm, neg_vectors_norm)
        
        # 统计信息
        avg_sim = np.mean(pos_neg_sim)
        sim_std = np.std(pos_neg_sim)
        sim_max = np.max(pos_neg_sim)
        sim_min = np.min(pos_neg_sim)
        
        print(f"\n数据特征分析结果:")
        print(f"1. 正负样本间相似度统计:")
        print(f"   平均相似度: {avg_sim:.4f}")
        print(f"   标准差: {sim_std:.4f}")
        print(f"   最大值: {sim_max:.4f}")
        print(f"   最小值: {sim_min:.4f}")
        
        # 判断是否需要平滑对比学习
        print(f"\n2. 对比学习策略建议:")
        if avg_sim > 0.3:
            print(f"   ✓ 平均相似度较高({avg_sim:.4f} > 0.3)，推荐使用平滑对比学习")
            print(f"   ✓ 建议温度系数: 1.0")
        elif avg_sim > 0.1:
            print(f"   × 平均相似度中等({avg_sim:.4f})，可尝试温度系数: 0.5-1.0")
        else:
            print(f"   ✓ 平均相似度较低({avg_sim:.4f})，可使用标准对比学习")
            print(f"   ✓ 建议温度系数: 0.07-0.2")
        
        # 计算正负样本内部相似度
        pos_sim = cosine_similarity(pos_vectors_norm)
        neg_sim = cosine_similarity(neg_vectors_norm)
        
        # 排除对角线
        np.fill_diagonal(pos_sim, 0)
        np.fill_diagonal(neg_sim, 0)
        
        avg_pos_intra_sim = np.mean(pos_sim)
        avg_neg_intra_sim = np.mean(neg_sim)
        
        print(f"\n3. 类内相似度统计:")
        print(f"   正样本内部平均相似度: {avg_pos_intra_sim:.4f}")
        print(f"   负样本内部平均相似度: {avg_neg_intra_sim:.4f}")
        
        return {
            'avg_cross_similarity': avg_sim,
            'cross_similarity_std': sim_std,
            'avg_pos_intra_similarity': avg_pos_intra_sim,
            'avg_neg_intra_similarity': avg_neg_intra_sim,
            'recommendation': 'smooth_contrastive' if avg_sim > 0.3 else 'standard_contrastive'
        }

    def optimize_batch_size(self, test_texts: List[str] = None, max_batch_size: int = 128):
        """
        自动优化批处理大小
        通过测试不同的批处理大小，找到最优的批处理大小
        
        Args:
            test_texts: 测试文本列表，如果为None则使用部分样本
            max_batch_size: 最大批处理大小
            
        Returns:
            int: 推荐的批处理大小
        """
        print("开始优化批处理大小...")
        
        if test_texts is None:
            # 使用部分样本进行测试（避免用全部数据）
            sample_size = min(100, len(self.data))
            test_texts = self.data['question_1'].tolist()[:sample_size]
        
        test_sizes = [1, 8, 16, 32, 64, 128]
        test_sizes = [size for size in test_sizes if size <= max_batch_size]
        
        results = []
        for batch_size in test_sizes:
            print(f"\n测试批处理大小: {batch_size}")
            
            import time
            start_time = time.time()
            
            # 使用批处理提取向量
            _ = self.extract_vectors_batch(test_texts, batch_size)
            
            elapsed_time = time.time() - start_time
            speed = len(test_texts) / elapsed_time  # 每秒处理的文本数
            
            print(f"处理时间: {elapsed_time:.2f}秒, 速度: {speed:.2f}文本/秒")
            results.append({'batch_size': batch_size, 'speed': speed, 'time': elapsed_time})
        
        # 找出速度最快的批处理大小
        best_result = max(results, key=lambda x: x['speed'])
        print(f"\n批处理大小优化结果:")
        for result in results:
            print(f"  批大小 {result['batch_size']}: {result['speed']:.2f}文本/秒 ({result['time']:.2f}秒)")
        
        print(f"\n推荐批处理大小: {best_result['batch_size']}")
        self.batch_size = best_result['batch_size']
        print(f"已更新批处理大小为: {self.batch_size}")
        
        return best_result['batch_size']


def get_prototype_vectors(csv_path: str, model_name: str = "t5-3b", 
                         n_positive_clusters: int = 10, n_negative_clusters: int = 10,
                         temperature: float = 1.0, save_path: str = None,
                         batch_size: int = 32) -> Dict:
    """
    构建价值观原型向量库的主函数
    
    Args:
        csv_path: 包含正负样本的CSV文件路径
        model_name: 预训练模型名称
        n_positive_clusters: 正样本聚类数量（默认10）
        n_negative_clusters: 负样本聚类数量（默认10）
        temperature: 温度系数（默认1.0，使用平滑对比）
        save_path: 可选的保存路径
        batch_size: 批处理大小（默认32）
        
    Returns:
        Dict: 包含原型和反原型的字典
    """
    print("开始构建价值观原型向量库...")
    
    # 加载T5模型
    print(f"加载模型: {model_name}")
    model = T5Model(model_name)
    
    # 创建原型学习器
    learner = PrototypeLearner(model, csv_path, batch_size=batch_size)
    
    # 分析数据特征
    characteristics = learner.analyze_data_characteristics()
    
    # 根据分析结果调整参数
    if characteristics['recommendation'] == 'smooth_contrastive':
        print("\n基于数据特征，使用平滑对比学习策略")
        temperature = 1.0
    
    # 构建原型空间
    prototypes = learner.build_prototype_space(
        n_positive_clusters=n_positive_clusters,
        n_negative_clusters=n_negative_clusters,
        temperature=temperature
    )
    
    # 评估原型质量
    quality_metrics, score = learner.evaluate_prototype_quality()
    
    # 如果提供了保存路径，则保存原型
    if save_path:
        learner.save_prototypes(save_path)
        # 同时保存聚类分配情况
        learner.save_cluster_assignments(save_path)
        
        # 保存分析报告
        report_path = os.path.join(save_path, learner.value_name, f"{learner.value_name}_analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'data_characteristics': characteristics,
                'quality_metrics': quality_metrics,
                'quality_score': score,
                'parameters': {
                    'temperature': temperature,
                    'n_positive_clusters': n_positive_clusters,
                    'n_negative_clusters': n_negative_clusters,
                    'batch_size': batch_size
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"分析报告已保存到: {report_path}")
    
    return prototypes


if __name__ == "__main__":
    # 示例：使用achievement_context_controlled.csv构建原型向量库
    # 构建相对于当前脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "..", "data", "controlled", "achievement_context_controlled.csv")
    save_path = os.path.join(script_dir, "prototypes")
    
    print("="*60)
    print("价值观原型向量库构建系统")
    print("="*60)
    
    # 加载T5模型
    print(f"加载模型: t5-3b")
    model = T5Model("t5-3b")
    
    # 创建原型学习器（设置批处理大小）
    learner = PrototypeLearner(model, csv_path, batch_size=32)
    
    # 可选：优化批处理大小（第一次运行时可以测试）
    # learner.optimize_batch_size()
    
    # 分析数据特征
    characteristics = learner.analyze_data_characteristics()
    
    # 根据分析结果确定参数范围
    if characteristics['avg_cross_similarity'] > 0.3:
        print("\n数据重叠度高，使用平滑对比学习（温度=1.0）")
        temperatures = [1.0]  # 只测试温度=1.0
    else:
        print("\n数据重叠度中等，测试多个温度系数")
        temperatures = [0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    
    clusters = [5, 7, 10, 15, 20]
    
    # 对比不同参数下的指标
    best_params, best_score, all_results = learner.compare_prototype_quality(temperatures, clusters, save_path)
    
    print(f"\n最佳参数组合: 温度系数={best_params['temperature']}, 聚类数={best_params['n_clusters']}")
    print(f"最佳得分: {best_score:.4f}")
    
    # 使用最佳参数构建最终原型空间
    print(f"\n使用最佳参数构建最终原型空间...")
    final_prototypes = learner.build_prototype_space(
        best_params['n_clusters'], 
        best_params['n_clusters'], 
        best_params['temperature']
    )
    
    # 保存最终原型
    learner.save_prototypes(save_path)
    learner.save_cluster_assignments(save_path)
    
    # 可视化最佳参数下的原型
    print("开始可视化最佳参数下的原型...")
    learner.visualize_prototypes(save_path=os.path.join(save_path, "visualization.png"))
    
    # 生成最终报告
    quality_metrics, score = learner.evaluate_prototype_quality()
    
    print(f"\n" + "="*60)
    print(f"{learner.value_name.capitalize()} 价值观原型构建完成!")
    print(f"最终质量得分: {score:.4f}")
    print(f"参数配置: 温度={best_params['temperature']}, 聚类数={best_params['n_clusters']}, 批大小={learner.batch_size}")
    print("="*60)