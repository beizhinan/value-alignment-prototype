import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from src.models.model import T5Model


class PrototypeRetriever(nn.Module):
    """
    原型检索器，一个神经网络，输入是目标句子 x，输出是每个候选原型被选中的概率
    """
    
    def __init__(self, embedding_dim, temperature=0.1):
        """
        初始化原型检索器
        
        Args:
            embedding_dim (int): 嵌入向量的维度
            temperature (float): 温度参数，控制分布的平滑程度
        """
        super(PrototypeRetriever, self).__init__()
        self.W = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        
    def forward(self, target_embedding, candidate_embeddings):
        """
        计算目标句子与候选原型的概率分布
        
        Args:
            target_embedding (torch.Tensor): 目标句子的嵌入向量，形状 [batch_size, embedding_dim]
            candidate_embeddings (torch.Tensor): 候选原型的嵌入向量，形状 [num_candidates, embedding_dim]
            
        Returns:
            torch.Tensor: 概率分布，形状 [batch_size, num_candidates]
        """
        # 线性变换候选原型嵌入
        transformed_candidates = self.W(candidate_embeddings)  # [num_candidates, embedding_dim]
        
        # 计算相似度得分
        scores = torch.matmul(target_embedding, transformed_candidates.t())  # [batch_size, num_candidates]
        
        # 应用温度参数
        scores = scores / self.temperature
        
        # 计算softmax概率分布
        probs = torch.softmax(scores, dim=-1)
        
        return probs


class InverseEditor(nn.Module):
    """
    逆编辑器，将原型和目标句子映射到编辑向量 z
    """
    
    def __init__(self, embedding_dim):
        """
        初始化逆编辑器
        
        Args:
            embedding_dim (int): 嵌入向量的维度
        """
        super(InverseEditor, self).__init__()
        self.edit_vector_generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, prototype_embedding, target_embedding):
        """
        生成编辑向量
        
        Args:
            prototype_embedding (torch.Tensor): 原型句子的嵌入向量
            target_embedding (torch.Tensor): 目标句子的嵌入向量
            
        Returns:
            torch.Tensor: 编辑向量 z
        """
        combined = torch.cat([prototype_embedding, target_embedding], dim=-1)
        edit_vector = self.edit_vector_generator(combined)
        return edit_vector


class Generator(nn.Module):
    """
    生成器，从原型和编辑向量生成目标句子
    """
    
    def __init__(self, embedding_dim):
        """
        初始化生成器
        
        Args:
            embedding_dim (int): 嵌入向量的维度
        """
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, prototype_embedding, edit_vector):
        """
        生成目标句子的嵌入
        
        Args:
            prototype_embedding (torch.Tensor): 原型句子的嵌入向量
            edit_vector (torch.Tensor): 编辑向量
            
        Returns:
            torch.Tensor: 生成的目标句子嵌入
        """
        combined = torch.cat([prototype_embedding, edit_vector], dim=-1)
        generated_embedding = self.generator(combined)
        return generated_embedding


class PrototypeLearner:
    """
    原型学习器，整合所有组件
    """
    
    def __init__(self, model_name="t5-small", temperature=0.1, max_prototypes=10000):
        """
        初始化原型学习器
        
        Args:
            model_name (str): T5模型名称
            temperature (float): 温度参数
            max_prototypes (int): 最大原型数量
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载T5模型作为嵌入编码器
        self.t5_model = T5Model(model_name=model_name)
        self.embedding_dim = self.t5_model.encoder_model.config.hidden_size
        self.temperature = temperature
        self.max_prototypes = max_prototypes
        
        # 初始化组件
        self.prototype_retriever = PrototypeRetriever(self.embedding_dim, temperature).to(self.device)
        self.inverse_editor = InverseEditor(self.embedding_dim).to(self.device)
        self.generator = Generator(self.embedding_dim).to(self.device)
        
        # 优化器
        self.retriever_optimizer = optim.Adam(self.prototype_retriever.parameters(), lr=1e-4)
        self.editor_optimizer = optim.Adam(list(self.inverse_editor.parameters()) + list(self.generator.parameters()), lr=1e-4)
        
        # 存储原型和嵌入
        self.prototype_texts = []  # 原型文本列表
        self.prototype_embeddings = None  # 原型嵌入张量 [num_prototypes, embedding_dim]
        
    def extract_embeddings(self, texts, batch_size=8):
        """
        使用T5模型提取文本嵌入
        
        Args:
            texts (list): 文本列表
            batch_size (int): 批次大小
            
        Returns:
            torch.Tensor: 嵌入向量张量
        """
        embeddings = []
        
        print(f"正在提取 {len(texts)} 条文本的嵌入...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # 分别处理每个文本
            for text in batch_texts:
                # 获取嵌入向量
                with torch.no_grad():
                    hidden_states = self.t5_model.get_encoder_last_hidden_state(text)
                    # 取序列的平均值作为句子嵌入
                    text_embedding = torch.mean(hidden_states[0], dim=0, keepdim=True)
                    embeddings.append(text_embedding)
                    
        return torch.cat(embeddings, dim=0).to(self.device)
    
    def build_candidate_prototype_pool(self, csv_files, sample_size_per_file=1000):
        """
        构建候选原型库
        对于大数据集，随机下采样一个子集作为初始原型库
        
        Args:
            csv_files (list): CSV文件路径列表
            sample_size_per_file (int): 每个文件的采样数量
        """
        all_texts = []
        
        print("构建候选原型库...")
        
        for csv_file in csv_files:
            print(f"处理文件: {os.path.basename(csv_file)}")
            
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 提取文本列（假设为question_1和question_2）
            text_cols = [col for col in df.columns if 'question' in col.lower()]
            
            if not text_cols:
                print(f"警告: 在文件 {csv_file} 中未找到question列")
                continue
                
            for col in text_cols:
                texts = df[col].dropna().tolist()
                
                # 如果文本太多，进行随机采样
                if len(texts) > sample_size_per_file:
                    sampled_indices = np.random.choice(len(texts), sample_size_per_file, replace=False)
                    texts = [texts[i] for i in sampled_indices]
                
                all_texts.extend(texts)
        
        print(f"总共收集到 {len(all_texts)} 条文本作为候选原型")
        
        # 限制总原型数量
        if len(all_texts) > self.max_prototypes:
            indices = np.random.choice(len(all_texts), self.max_prototypes, replace=False)
            all_texts = [all_texts[i] for i in indices]
            print(f"随机采样后保留 {len(all_texts)} 条候选原型")
        
        # 更新原型文本列表
        self.prototype_texts = all_texts
        
        # 提取所有候选原型的嵌入
        self.prototype_embeddings = self.extract_embeddings(all_texts)
        
        print(f"候选原型库构建完成，共 {len(self.prototype_texts)} 个候选原型")
    
    def compute_generation_loss(self, generated_embedding, target_embedding):
        """
        计算生成损失（使用余弦相似度的负值）
        
        Args:
            generated_embedding (torch.Tensor): 生成的嵌入
            target_embedding (torch.Tensor): 目标嵌入
            
        Returns:
            torch.Tensor: 生成损失
        """
        # 计算余弦相似度
        cos_sim = nn.functional.cosine_similarity(generated_embedding, target_embedding, dim=-1)
        # 返回负的余弦相似度作为损失
        return -cos_sim.mean()
    
    def train_step(self, target_texts):
        """
        单步训练
        
        Args:
            target_texts (list): 目标文本列表
        """
        # 提取目标文本的嵌入
        target_embeddings = self.extract_embeddings(target_texts)
        
        # 禁止选择自身作为原型（这里我们简化处理，实际上需要在计算概率时排除当前样本）
        # 但在当前实现中，我们假设target_texts和self.prototype_texts不同
        
        # 获取原型检索概率
        prototype_probs = self.prototype_retriever(target_embeddings, self.prototype_embeddings)
        
        # 选择最有可能的原型（简化：选择top-1）
        _, top_indices = torch.topk(prototype_probs, k=1, dim=1)
        selected_indices = top_indices.squeeze(1)
        
        # 获取选中的原型嵌入
        selected_prototype_embeddings = self.prototype_embeddings[selected_indices]
        
        # 使用逆编辑器生成编辑向量
        edit_vectors = self.inverse_editor(selected_prototype_embeddings, target_embeddings)
        
        # 使用生成器生成目标嵌入
        generated_embeddings = self.generator(selected_prototype_embeddings, edit_vectors)
        
        # 计算生成损失
        generation_loss = self.compute_generation_loss(generated_embeddings, target_embeddings)
        
        # 反向传播更新编辑器和生成器
        self.editor_optimizer.zero_grad()
        generation_loss.backward(retain_graph=True)
        self.editor_optimizer.step()
        
        # 为了更新原型检索器，我们需要使用REINFORCE或类似的方法
        # 这里我们使用一个简化的近似方法：基于生成质量更新原型选择概率
        with torch.no_grad():
            quality_scores = nn.functional.cosine_similarity(generated_embeddings, target_embeddings, dim=-1)
        
        # 使用REINFORCE更新原型检索器
        log_probs = torch.log(torch.gather(prototype_probs, 1, selected_indices.unsqueeze(1)).squeeze(1) + 1e-8)
        reinforce_loss = -(log_probs * quality_scores.detach()).mean()
        
        self.retriever_optimizer.zero_grad()
        reinforce_loss.backward()
        self.retriever_optimizer.step()
        
        return generation_loss.item(), reinforce_loss.item()
    
    def train(self, epochs=10, batch_size=16, target_csv_files=None):
        """
        训练原型学习器
        
        Args:
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            target_csv_files (list): 目标CSV文件列表
        """
        print("开始训练原型学习器...")
        
        # 如果没有指定目标文件，则使用原型库中的文本
        if target_csv_files is None:
            target_texts = self.prototype_texts
        else:
            target_texts = []
            for csv_file in target_csv_files:
                df = pd.read_csv(csv_file)
                text_cols = [col for col in df.columns if 'question' in col.lower()]
                for col in text_cols:
                    target_texts.extend(df[col].dropna().tolist())
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # 随机打乱目标文本
            np.random.shuffle(target_texts)
            
            epoch_gen_loss = 0
            epoch_reinforce_loss = 0
            batches_processed = 0
            
            for i in tqdm(range(0, len(target_texts), batch_size)):
                batch_texts = target_texts[i:i+batch_size]
                
                gen_loss, reinforce_loss = self.train_step(batch_texts)
                
                epoch_gen_loss += gen_loss
                epoch_reinforce_loss += reinforce_loss
                batches_processed += 1
            
            avg_gen_loss = epoch_gen_loss / batches_processed
            avg_reinforce_loss = epoch_reinforce_loss / batches_processed
            
            print(f"Epoch {epoch+1} - 生成损失: {avg_gen_loss:.4f}, 强化学习损失: {avg_reinforce_loss:.4f}")
    
    def sparse_prototype_selection(self, top_k=2000, cumulative_prob_threshold=0.9):
        """
        稀疏化原型选择：从全库中选择最重要的原型
        
        Args:
            top_k (int): 选择的原型数量上限
            cumulative_prob_threshold (float): 累积概率阈值
        """
        print("开始稀疏化原型选择...")
        
        # 计算每个原型的重要性分数（这里我们使用一些启发式方法）
        # 方法1：计算原型与其他原型的平均相似度的倒数
        with torch.no_grad():
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(
                self.prototype_embeddings.cpu().numpy()
            )
            
            # 对角线元素设为0（不计算与自身的相似度）
            np.fill_diagonal(similarity_matrix, 0)
            
            # 计算每个原型与其他原型的平均相似度
            avg_similarities = np.mean(similarity_matrix, axis=1)
            
            # 重要性分数 = 1 / (平均相似度 + epsilon)
            epsilon = 1e-8
            importance_scores = 1 / (avg_similarities + epsilon)
        
        # 归一化重要性分数
        importance_scores = importance_scores / np.sum(importance_scores)
        
        # 按重要性排序
        sorted_indices = np.argsort(-importance_scores)
        
        # 选择top-k原型，或者直到累积概率达到阈值
        selected_indices = []
        cum_prob = 0
        
        for idx in sorted_indices[:top_k]:
            selected_indices.append(idx)
            cum_prob += importance_scores[idx]
            
            if cum_prob >= cumulative_prob_threshold:
                print(f"选择前 {len(selected_indices)} 个原型，累积概率达到 {cum_prob:.3f}")
                break
        
        # 更新原型库
        self.prototype_texts = [self.prototype_texts[i] for i in selected_indices]
        self.prototype_embeddings = self.prototype_embeddings[selected_indices]
        
        print(f"稀疏化后剩余 {len(self.prototype_texts)} 个原型")
        
        return self.prototype_texts, self.prototype_embeddings
    
    def save_prototypes(self, output_dir, value_name="merged"):
        """
        保存原型到文件
        
        Args:
            output_dir (str): 输出目录
            value_name (str): 价值观名称
        """
        print(f"保存原型到 {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原型文本
        prototypes_file = os.path.join(output_dir, f"{value_name}_prototypes.json")
        with open(prototypes_file, 'w', encoding='utf-8') as f:
            json.dump({
                "prototypes": self.prototype_texts,
                "count": len(self.prototype_texts)
            }, f, ensure_ascii=False, indent=2)
        
        # 保存原型嵌入（使用pickle格式）
        embeddings_file = os.path.join(output_dir, f"{value_name}_embeddings.pkl")
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.prototype_embeddings.cpu().numpy(), f)
        
        print(f"原型已保存到 {prototypes_file}")
        print(f"嵌入已保存到 {embeddings_file}")


def main():
    """主函数，执行完整的原型构建流程"""
    print("开始构建原型库...")
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 查找合并后的数据集
    merged_data_dir = os.path.join(project_root, "data", "merged")
    
    if not os.path.exists(merged_data_dir):
        print(f"错误: 目录 {merged_data_dir} 不存在")
        return
    
    # 获取所有合并后的CSV文件
    csv_files = [os.path.join(merged_data_dir, f) for f in os.listdir(merged_data_dir) 
                 if f.endswith('_merged.csv')]
    
    if not csv_files:
        print(f"错误: 在 {merged_data_dir} 中未找到_merged.csv文件")
        return
    
    print(f"找到 {len(csv_files)} 个合并后的数据集文件")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # 初始化原型学习器
    learner = PrototypeLearner(model_name="t5-3b", temperature=0.1, max_prototypes=5000)
    
    # 1. 构建候选原型库
    learner.build_candidate_prototype_pool(csv_files, sample_size_per_file=1000)
    
    # 2. 训练原型检索器和编辑器
    learner.train(epochs=5, batch_size=8, target_csv_files=csv_files)
    
    # 3. 稀疏化原型选择
    selected_prototypes, selected_embeddings = learner.sparse_prototype_selection(top_k=2000, cumulative_prob_threshold=0.9)
    
    # 4. 保存最终原型
    output_dir = os.path.join(project_root, "prototypes")
    learner.save_prototypes(output_dir, value_name="merged")
    
    print("原型库构建完成！")


if __name__ == "__main__":
    main()