import json
import os
import logging
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
import torch
from transformers import AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# 检查是否有GPU可用
cuda_available = torch.cuda.is_available()

# 加载训练数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 将数据转换为NER模型所需的格式，使用分词器处理
def convert_to_ner_format(data, print_data=False):
    # 加载多语言分词器
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
    
    ner_data = []
    
    for item_idx, item in enumerate(data):
        text = item['text']
        labels = item['label']

        if print_data:
            print(text)
        
        # 创建一个字符级别的标签列表，初始化为 'O'（表示不是实体）
        char_labels = ['O'] * len(text)
        
        # 根据标注信息填充标签 - 按照实体类型的优先级排序处理
        # 优先级顺序：EP > SEASON > GROUP > LANG > RES > TITLE
        priority_order = {
            'EP': 1,
            'SEASON': 2,
            'GROUP': 3,
            'LANG': 4,
            'RES': 5,
            'TITLE': 6
        }
        
        # 按优先级排序标签
        sorted_labels = sorted(labels, key=lambda x: priority_order.get(x['labels'][0], 999))
        
        # 处理标签
        for label_info in sorted_labels:
            start = label_info['start']
            end = label_info['end']
            entity_type = label_info['labels'][0]  # 取第一个标签
            
            # 使用BIO标注方案：B-表示实体开始，I-表示实体内部
            char_labels[start] = f'B-{entity_type}'
            for i in range(start + 1, end):
                char_labels[i] = f'I-{entity_type}'
        
        # 使用分词器进行分词，并获取偏移量映射
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offset_mapping = encoding['offset_mapping']
        
        # 为每个token分配标签
        token_labels = []
        for i, (start_offset, end_offset) in enumerate(offset_mapping):
            # 如果是特殊token，如[CLS], [SEP]等
            if start_offset == end_offset:
                token_labels.append('O')
                continue
                
            # 获取token对应的字符级标签
            # 我们采用多数投票的方式确定token的标签
            label_counts = {}
            for j in range(start_offset, end_offset):
                label = char_labels[j]
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            
            # 找出出现次数最多的标签
            max_count = 0
            majority_label = 'O'
            for label, count in label_counts.items():
                if count > max_count:
                    max_count = count
                    majority_label = label
            
            # 如果多数标签是I-，但前一个token不是相同类型的B-或I-，则改为B-
            if majority_label.startswith('I-'):
                entity_type = majority_label[2:]
                if i == 0 or not token_labels[i-1].endswith(entity_type):
                    majority_label = f'B-{entity_type}'
            
            token_labels.append(majority_label)
        
        # 将token和标签组合成训练数据格式
        words_and_labels = []
        for i, token in enumerate(tokens):
            words_and_labels.append([item_idx, token, token_labels[i]])
            if print_data:
                print(item_idx, token, token_labels[i])
        
        ner_data.extend(words_and_labels)
    
    # 转换为DataFrame，添加sentence_id列
    df = pd.DataFrame(ner_data, columns=['sentence_id', 'words', 'labels'])
    return df

# 主函数
def main():
    # 加载数据 - 使用新生成的数据
    data = load_data('data/anime_ner_training_data.json')
    print(f"加载了 {len(data)} 条训练数据")
    
    # 转换数据格式
    df = convert_to_ner_format(data)
    print(f"转换后的数据形状: {df.shape}")
    print(df.head())  # 只打印前几行，而不是整个DataFrame
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:]
    
    train_df = convert_to_ner_format(train_data)
    eval_df = convert_to_ner_format(eval_data)
    
    print(f"训练集大小: {len(train_data)} 条数据")
    print(f"验证集大小: {len(eval_data)} 条数据")
    
    # 定义模型参数
    model_args = NERArgs()
    model_args.train_batch_size = 8  # 减小批次大小以减少内存使用
    model_args.eval_batch_size = 8
    model_args.num_train_epochs = 10  # 减少训练轮数以加快训练
    model_args.learning_rate = 3e-5  # 调整学习率
    model_args.overwrite_output_dir = True
    model_args.save_steps = 1000  # 每1000步保存一次
    model_args.save_model_every_epoch = True
    model_args.output_dir = "outputs/"
    model_args.best_model_dir = "outputs/best_model/"
    model_args.evaluate_during_training = True
    model_args.use_early_stopping = True
    model_args.early_stopping_patience = 2  # 减少早停耐心值以加快训练
    model_args.early_stopping_delta = 0.01
    model_args.max_seq_length = 128
    model_args.manual_seed = 42
    model_args.fp16 = False  # 禁用混合精度训练，提高稳定性
    model_args.gradient_accumulation_steps = 2  # 梯度累积，有效增加批次大小
    model_args.weight_decay = 0.01  # 添加权重衰减，减少过拟合
    model_args.warmup_ratio = 0.1  # 添加预热阶段，提高训练稳定性
    
    # 创建模型
    # 使用多语言预训练模型
    model = NERModel(
        "bert", 
        "bert-base-multilingual-cased",
        args=model_args,
        labels=["O", "B-TITLE", "I-TITLE", "B-GROUP", "I-GROUP", "B-EP", "I-EP", 
                "B-SEASON", "I-SEASON", "B-RES", "I-RES", "B-LANG", "I-LANG"],
        use_cuda=cuda_available,
    )
    
    # 训练模型
    model.train_model(train_df, eval_df=eval_df)
    
    # 评估模型
    result, model_outputs, predictions = model.eval_model(eval_df)
    print("评估结果:", result)
    
    print("模型训练和评估完成！")

if __name__ == "__main__":
    main() 