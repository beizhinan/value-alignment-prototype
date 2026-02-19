import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel

class T5Model:
    """
    T5模型类，用于下载和使用预训练的T5模型
    """
    
    def __init__(self, model_name="t5-small", model_dir=None):
        """
        初始化T5模型
        
        Args:
            model_name (str): 预训练模型的名称，如"t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
            model_dir (str): 模型保存的目录，如果为None则使用默认缓存目录
        """
        self.model_name = model_name
        self.model_dir = model_dir
        
        # 加载预训练的T5模型和分词器
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=model_dir
        )
        
        # 加载T5编码器模型
        self.encoder_model = T5EncoderModel.from_pretrained(
            model_name,
            cache_dir=model_dir
        )
        
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=model_dir
        )
        
        print(f"T5模型 {model_name} 加载成功")
        
    def generate(self, input_text, max_length=512, num_beams=4, early_stopping=True, **kwargs):
        """
        使用T5模型生成文本
        
        Args:
            input_text (str): 输入文本
            max_length (int): 生成文本的最大长度
            num_beams (int): beam search的数量
            early_stopping (bool): 是否提前停止
            **kwargs: 其他生成参数
            
        Returns:
            str: 生成的文本
        """
        # 对输入文本进行编码
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=max_length,
            truncation=True
        )
        
        # 生成输出
        with torch.no_grad():  # 禁用梯度计算以节省内存
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                **kwargs
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def get_encoder_last_hidden_state(self, input_text, return_tokens=False):
        """
        获取T5编码器最后一层的隐藏状态（嵌入向量）
        
        Args:
            input_text (str): 输入文本
            return_tokens (bool): 是否同时返回token IDs
            
        Returns:
            torch.Tensor or tuple: 编码器最后一层的隐藏状态向量，
                           如果return_tokens为True则返回(隐藏状态, token_ids)
        """
        # 对输入文本进行编码
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # 使用编码器获取最后一层的隐藏状态
        with torch.no_grad():  # 冻结参数，不进行梯度计算
            outputs = self.encoder_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 获取最后一层的隐藏状态
            last_hidden_states = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]
        
        if return_tokens:
            return last_hidden_states, input_ids
        else:
            return last_hidden_states
    
    def save_model(self, save_dir):
        """
        保存模型到指定目录
        
        Args:
            save_dir (str): 保存模型的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"模型已保存到: {save_dir}")


def download_pretrained_t5(model_name="t5-small", save_dir=None):
    """
    下载预训练的T5模型
    
    Args:
        model_name (str): 预训练模型的名称
        save_dir (str): 保存模型的目录
        
    Returns:
        T5Model: T5模型实例
    """
    print(f"开始下载预训练T5模型: {model_name}")
    
    # 创建T5模型实例，这会自动下载预训练模型
    t5_model = T5Model(model_name, model_dir=save_dir)
    
    print(f"预训练T5模型 {model_name} 下载完成")
    
    return t5_model


if __name__ == "__main__":
    # 示例：下载并使用t5-small模型
    print("开始下载预训练T5模型...")
    
    # 下载t5-3b模型
    model = download_pretrained_t5("t5-3b")
    print("下载完成...")