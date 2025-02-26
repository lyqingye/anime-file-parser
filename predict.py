import json
import os
import logging
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
import torch
from transformers import AutoTokenizer
import requests
from urllib.parse import urlparse

# 设置日志
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# 检查是否有GPU可用
cuda_available = torch.cuda.is_available()

# 定义标签列表
labels = ["O", "B-TITLE", "I-TITLE", "B-GROUP", "I-GROUP", "B-EP", "I-EP", 
          "B-SEASON", "I-SEASON", "B-RES", "I-RES", "B-LANG", "I-LANG"]

def tokenize_filename(filename):
    """
    将文件名分割成标记（tokens）
    """
    # 移除文件扩展名
    filename = os.path.splitext(filename)[0]
    
    # 替换一些特殊字符为空格，以便更好地分割
    for char in "()[]{}【】":
        filename = filename.replace(char, f" {char} ")
    
    # 在下划线、连字符和斜杠周围添加空格
    for char in "_-/+":
        filename = filename.replace(char, f" {char} ")
    
    # 分割成标记并过滤掉空标记
    tokens = [token for token in filename.split() if token]
    
    return tokens

# 从HTTP链接加载数据
def load_data_from_url(url):
    """
    从HTTP链接加载JSON数据
    
    Args:
        url (str): 指向JSON数据的HTTP链接
        
    Returns:
        list: 加载的数据
        
    Raises:
        Exception: 如果请求失败或数据格式不正确
    """
    try:
        # 检查URL是否有效
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"无效的URL: {url}")
        
        # 发送HTTP请求
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 如果请求失败，抛出异常
        
        # 尝试解析JSON数据
        data = response.json()
        
        print(f"成功从 {url} 加载了数据")
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"HTTP请求错误: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        raise
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

# 加载模型
model = NERModel(
    "bert",
    "outputs/best_model",
    labels=labels,
    use_cuda=cuda_available,
)

def process_predictions(predictions):
    """
    处理预测结果，将其转换为JSON格式
    每个标签对应的所有值都会被合并
    """
    result = {}
    
    # 初始化结果字典
    for label in labels:
        if label != "O":  # 忽略O标签
            # 去掉B-和I-前缀
            clean_label = label[2:] if label.startswith("B-") or label.startswith("I-") else label
            result[clean_label] = []
    
    # 处理预测结果
    for sentence in predictions:
        current_entity = None
        current_text = ""
        
        for token_dict in sentence:
            for token, label in token_dict.items():
                # 如果是O标签，结束当前实体的收集
                if label == "O":
                    if current_entity and current_text:
                        result[current_entity].append(current_text.strip())
                        current_entity = None
                        current_text = ""
                else:
                    # 提取标签名称（去掉B-和I-前缀）
                    entity_type = label[2:]
                    
                    # 如果是B-开头，开始一个新实体
                    if label.startswith("B-"):
                        # 如果之前有未完成的实体，先保存
                        if current_entity and current_text:
                            result[current_entity].append(current_text.strip())
                        
                        current_entity = entity_type
                        current_text = token
                    # 如果是I-开头，继续当前实体
                    elif label.startswith("I-"):
                        # 如果当前没有实体或实体类型不匹配，则视为新实体
                        if not current_entity:
                            current_entity = entity_type
                            current_text = token
                        elif current_entity == entity_type:
                            current_text += " " + token
                        else:
                            # 实体类型变化，保存之前的实体并开始新实体
                            if current_text:
                                result[current_entity].append(current_text.strip())
                            current_entity = entity_type
                            current_text = token
        
        # 处理最后一个实体
        if current_entity and current_text:
            result[current_entity].append(current_text.strip())
    
    # 移除空列表
    result = {k: v for k, v in result.items() if v}
    
    return result

def process_file(input_file, output_file):
    """
    处理输入文件中的每一行，并将结果保存为JSON文件
    
    Args:
        input_file (str): 输入文件路径，每行一个文件名
        output_file (str): 输出JSON文件路径
    """
    results = []
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        filenames = [line.strip() for line in f if line.strip()]
    
    print(f"读取到 {len(filenames)} 个文件名")
    
    # 处理每个文件名
    for i, filename in enumerate(filenames):
        print(f"处理第 {i+1}/{len(filenames)} 个文件名: {filename}")
        
        # 预测
        tokens = tokenize_filename(filename)
        predictions, _ = model.predict([" ".join(tokens)])
        
        # 处理预测结果
        processed_result = process_predictions(predictions)
        
        # 添加到结果列表
        results.append({
            "original_filename": filename,
            "parsed_result": processed_result
        })
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {output_file}")
    return results

def process_url(url, output_file):
    """
    处理从URL加载的文件名列表，并将结果保存为JSON文件
    
    Args:
        url (str): 指向JSON数据的HTTP链接，数据应为文件名列表
        output_file (str): 输出JSON文件路径
    """
    # 从URL加载数据
    data = load_data_from_url(url)
    
    # 确保数据是列表格式
    if isinstance(data, dict) and "filenames" in data:
        filenames = data["filenames"]
    elif isinstance(data, list):
        filenames = data
    else:
        raise ValueError("URL返回的数据格式不正确，应为文件名列表或包含filenames字段的对象")
    
    results = []
    print(f"从URL加载了 {len(filenames)} 个文件名")
    
    # 处理每个文件名
    for i, filename in enumerate(filenames):
        if not isinstance(filename, str):
            print(f"跳过非字符串项: {filename}")
            continue
            
        print(f"处理第 {i+1}/{len(filenames)} 个文件名: {filename}")
        
        # 预测
        tokens = tokenize_filename(filename)
        predictions, _ = model.predict([" ".join(tokens)])
        
        # 处理预测结果
        processed_result = process_predictions(predictions)
        
        # 添加到结果列表
        results.append({
            "original_filename": filename,
            "parsed_result": processed_result
        })
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {output_file}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='预测动漫文件名中的实体')
    parser.add_argument('--text', type=str, help='要预测的文件名')
    parser.add_argument('--input_file', type=str, help='输入文件路径，每行一个文件名')
    parser.add_argument('--input_url', type=str, help='输入数据的HTTP链接')
    parser.add_argument('--output_file', type=str, default='parsed_results.json', help='输出JSON文件路径')
    args = parser.parse_args()
    
    if args.input_url:
        # 处理URL
        process_url(args.input_url, args.output_file)
    elif args.input_file:
        # 处理文件
        process_file(args.input_file, args.output_file)
    elif args.text:
        # 处理单个文件名
        tokens = tokenize_filename(args.text)
        predictions, _ = model.predict([" ".join(tokens)])
        
        # 打印原始预测结果
        print("原始预测结果:")
        print(predictions)
        
        # 处理并打印JSON格式的结果
        processed_result = process_predictions(predictions)
        print("\nJSON格式结果:")
        print(json.dumps(processed_result, ensure_ascii=False, indent=2))
    else:
        parser.error("请提供 --text、--input_file 或 --input_url 参数")