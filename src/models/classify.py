import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from transformers import T5EncoderModel, T5Tokenizer
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到sys.path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model import T5Model


class ValueAlignmentClassifier:
    """
    价值观对齐分类器
    使用T5模型提取文本特征，训练先进分类器
    """
    
    def __init__(self):
        # 初始化T5模型
        self.t5_model = T5Model()
        self.scaler = StandardScaler()
        self.classifiers = {}  # 存储多个模型 {'value_type': {'rf': rf_model, 'mlp': mlp_model}}
        self.value_types = []
        
    def extract_features(self, texts):
        """
        使用T5模型提取文本特征
        
        Args:
            texts (list): 文本列表
            
        Returns:
            numpy.ndarray: 特征矩阵
        """
        features = []
        
        for i, text in enumerate(texts):
            if i % 50 == 0:  # 每50个文本打印一次进度
                print(f"正在提取特征: {i}/{len(texts)}")
                
            # 获取T5编码器的隐藏状态
            with torch.no_grad():
                encoded = self.t5_model.get_encoder_last_hidden_state(text)
                # 使用均值池化得到固定长度的向量
                pooled = torch.mean(encoded, dim=1).squeeze(0)
                features.append(pooled.numpy())
                
        return np.array(features)
    
    def prepare_data(self, csv_path):
        """
        准备训练数据
        
        Args:
            csv_path (str): CSV文件路径
            
        Returns:
            tuple: (features, labels)
        """
        print(f"正在加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 提取文本并创建标签
        # question_1 是正原型 (标签 0)，question_2 是负原型 (标签 1)
        texts_q1 = df['question_1'].tolist()
        texts_q2 = df['question_2'].tolist()
        
        all_texts = texts_q1 + texts_q2
        labels = [0] * len(texts_q1) + [1] * len(texts_q2)  # 0: 正原型, 1: 负原型
        
        # 提取特征
        features = self.extract_features(all_texts)
        
        return features, np.array(labels)
    
    def train_for_value_type(self, csv_path, value_name):
        """
        训练特定价值观类型的分类器
        
        Args:
            csv_path (str): CSV文件路径
            value_name (str): 价值观类型名称
        """
        print(f"\n正在训练 {value_name} 分类器...")
        
        # 准备数据
        features, labels = self.prepare_data(csv_path)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练多层感知机分类器
        print(f"训练多层感知机分类器...")
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        mlp_classifier.fit(X_train_scaled, y_train)
        y_pred_mlp = mlp_classifier.predict(X_test_scaled)
        accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
        
        print(f"多层感知机分类器准确率: {accuracy_mlp:.4f}")
        print(f"多层感知机分类报告:\n{classification_report(y_test, y_pred_mlp, target_names=['正原型', '负原型'])}")
        
        # 保存分类器
        self.classifiers[value_name] = {
            'models': {'mlp': mlp_classifier},
            'best_model': 'mlp'
        }
        self.value_types.append(value_name)
        
        return accuracy_mlp
    
    def predict_negative_probability(self, text, value_type, model_type='best'):
        """
        预测输入文本为负原型的概率
        
        Args:
            text (str): 输入文本
            value_type (str): 价值观类型
            model_type (str): 模型类型 ('best', 'mlp')
            
        Returns:
            float: 为负原型的概率
        """
        if value_type not in self.classifiers:
            raise ValueError(f"未找到 {value_type} 的分类器，请先训练该分类器")
        
        # 提取特征
        feature = self.t5_model.get_encoder_last_hidden_state(text)
        feature = torch.mean(feature, dim=1).squeeze(0).numpy().reshape(1, -1)
        
        # 标准化特征
        feature_scaled = self.scaler.transform(feature)
        
        # 根据model_type选择模型进行预测
        classifier_group = self.classifiers[value_type]
        
        if model_type == 'best' or model_type == 'mlp':
            model = classifier_group['models']['mlp']
        else:
            raise ValueError(f"未知的模型类型: {model_type}. 可选值: 'best', 'mlp'")
        
        # 预测概率
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_scaled)
            # 返回负原型（标签1）的概率
            return probabilities[0][1]
        else:
            # 如果模型没有predict_proba方法，则返回预测结果
            prediction = model.predict(feature_scaled)[0]
            return float(prediction)
    
    def predict_with_all_models(self, text, value_type):
        """
        使用所有模型预测输入文本为负原型的概率
        
        Args:
            text (str): 输入文本
            value_type (str): 价值观类型
            
        Returns:
            dict: 包含各种模型预测结果的字典
        """
        if value_type not in self.classifiers:
            raise ValueError(f"未找到 {value_type} 的分类器，请先训练该分类器")
        
        # 提取特征
        feature = self.t5_model.get_encoder_last_hidden_state(text)
        feature = torch.mean(feature, dim=1).squeeze(0).numpy().reshape(1, -1)
        
        # 标准化特征
        feature_scaled = self.scaler.transform(feature)
        
        results = {}
        classifier_group = self.classifiers[value_type]['models']
        
        # 只有一个模型（MLP），所以直接预测
        model = classifier_group['mlp']
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_scaled)
            results['mlp'] = probabilities[0][1]  # 负原型概率
        else:
            prediction = model.predict(feature_scaled)[0]
            results['mlp'] = float(prediction)
        
        # 由于只有一个模型，平均值就是这个模型的值
        results['average'] = results['mlp']
        
        return results
    
    def save_models(self, save_dir):
        """
        保存所有训练好的模型
        
        Args:
            save_dir (str): 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存分类器组
        for value_type, classifier_group in self.classifiers.items():
            group_path = os.path.join(save_dir, f"{value_type}_classifiers")
            os.makedirs(group_path, exist_ok=True)
            
            # 保存各个模型
            for model_name, model in classifier_group['models'].items():
                model_path = os.path.join(group_path, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)
                print(f"已保存 {value_type} 的 {model_name} 模型到: {model_path}")
            
            # 保存最佳模型信息
            best_model_info_path = os.path.join(group_path, "best_model.txt")
            with open(best_model_info_path, 'w') as f:
                f.write(f"{classifier_group['best_model']}\n")
        
        # 保存标准化器
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"已保存标准化器到: {scaler_path}")
        
        # 保存价值观类型列表
        types_path = os.path.join(save_dir, "value_types.txt")
        with open(types_path, 'w') as f:
            for value_type in self.value_types:
                f.write(f"{value_type}\n")
        print(f"已保存价值观类型列表到: {types_path}")
    
    def load_models(self, load_dir):
        """
        加载已训练的模型
        
        Args:
            load_dir (str): 加载目录
        """
        # 加载标准化器
        scaler_path = os.path.join(load_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # 加载价值观类型列表
        types_path = os.path.join(load_dir, "value_types.txt")
        with open(types_path, 'r') as f:
            self.value_types = [line.strip() for line in f.readlines()]
        
        # 加载各个分类器组
        for value_type in self.value_types:
            group_path = os.path.join(load_dir, f"{value_type}_classifiers")
            
            classifier_group = {'models': {}, 'best_model': None}
            
            # 加载MLP模型
            model_path = os.path.join(group_path, "mlp_model.pkl")
            if os.path.exists(model_path):
                classifier_group['models']['mlp'] = joblib.load(model_path)
                print(f"已加载 {value_type} 的 MLP 模型")
            
            # 加载最佳模型信息
            best_model_info_path = os.path.join(group_path, "best_model.txt")
            if os.path.exists(best_model_info_path):
                with open(best_model_info_path, 'r') as f:
                    classifier_group['best_model'] = f.read().strip()
            
            self.classifiers[value_type] = classifier_group


def train_all_classifiers(data_dir, save_dir):
    """
    训练所有价值观类型的分类器
    
    Args:
        data_dir (str): 数据目录
        save_dir (str): 模型保存目录
    """
    classifier = ValueAlignmentClassifier()
    
    # 获取所有价值观数据集
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_merged.csv')]
    
    print(f"发现 {len(csv_files)} 个价值观数据集: {csv_files}")
    
    for csv_file in csv_files:
        # 提取价值观类型名称（例如：'achievement_merged.csv' -> 'achievement'）
        value_name = csv_file.replace('_merged.csv', '')
        
        # 构建完整路径
        csv_path = os.path.join(data_dir, csv_file)
        
        # 训练分类器
        try:
            classifier.train_for_value_type(csv_path, value_name)
        except Exception as e:
            print(f"训练 {value_name} 分类器时出错: {e}")
            continue
    
    # 保存所有模型
    classifier.save_models(save_dir)
    
    print(f"\n所有分类器训练完成，已保存到: {save_dir}")
    
    return classifier


def main():
    """
    主函数
    """
    # 设置数据和模型保存路径
    data_dir = os.path.join(project_root, "data", "merged")
    save_dir = os.path.join(project_root, "models", "value_alignment_classifiers")
    
    # 训练所有分类器
    classifier = train_all_classifiers(data_dir, save_dir)
    
    # 测试示例文本
    print("\n" + "="*60)
    print("测试给定文本属于负例的概率:")
    
    test_text = "I wake up every morning driven by the desire to push my limits, achieve something meaningful, and prove to myself that I am capable of greatness."
    print(f"测试文本: {test_text}")
    print("-"*60)
    
    # 对所有价值观类型进行测试
    for value_type in classifier.value_types:
        try:
            # 使用最佳模型预测
            neg_prob_best = classifier.predict_negative_probability(test_text, value_type, model_type='best')
            print(f"{value_type:15}价值观 (最佳模型): {neg_prob_best:.4f}")
            
            # 使用所有模型预测
            all_predictions = classifier.predict_with_all_models(test_text, value_type)
            print(f"  - 多层感知机:   {all_predictions.get('mlp', 0):.4f}")
            print(f"  - 平均概率:     {all_predictions.get('average', 0):.4f}")
            print()
        except Exception as e:
            print(f"{value_type:15}价值观: 计算出错 - {e}")


if __name__ == "__main__":
    main()