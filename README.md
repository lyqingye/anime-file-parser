# 动漫文件名命名实体识别

该项目使用 SimpleTransformers 训练一个模型，用于从动漫文件名中识别各种实体，如番剧名、字幕组、集数等。

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── data/                     # 数据目录
│   └── train_data.json       # 训练数据（从label studio导出）
├── train.py                  # 训练脚本
├── predict.py                # 预测脚本
├── test.py                   # 测试脚本
├── test_samples.txt          # 测试样本
├── outputs/                  # 模型输出目录（训练后生成）
│   └── best_model/           # 最佳模型目录
├── predictions.json          # 预测结果
└── venv/                     # Python虚拟环境
```

## 环境设置

本项目使用 Python 虚拟环境管理依赖。按照以下步骤设置环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## NER 标签说明

本项目识别以下实体类型：

- 番剧名: TITLE
- 字幕组: GROUP
- 集数: EP
- 季数: SEASON
- 分辨率: RES
- 字幕语言: LANG

## 训练模型

使用以下命令训练模型：

```bash
python train.py
```

训练过程会自动：

1. 加载并处理`data/train_data.json`中的数据
2. 将数据分为训练集和验证集
3. 训练 BERT 模型进行命名实体识别
4. 在验证集上评估模型性能
5. 将最佳模型保存到`outputs/best_model/`目录

## 使用模型进行预测

训练完成后，可以使用以下命令进行预测：

### 预测单个文本

```bash
python predict.py --text "[喵萌奶茶屋&LoliHouse] 青箱 / Ao no Hako / Blue Box - 19 [WebRip 1080p HEVC-10bit AAC][简繁日内封字幕]"
```

### 预测多个文本（从文件）

```bash
python predict.py --input_file test_samples.txt --output_file predictions.json
```

预测结果将保存为 JSON 格式，包含识别出的各类实体及其位置信息。

### 测试单个文本并查看详细结果

```bash
python test.py --text "[喵萌奶茶屋&LoliHouse] 青箱 / Ao no Hako / Blue Box - 19 [WebRip 1080p HEVC-10bit AAC][简繁日内封字幕]"
```

## 预测结果示例

```json
{
  "text": "[喵萌奶茶屋&LoliHouse] 青箱 / Ao no Hako / Blue Box - 19 [WebRip 1080p HEVC-10bit AAC][简繁日内封字幕]",
  "entities": {
    "TITLE": [
      {
        "text": "青箱",
        "start": 18,
        "end": 20
      },
      {
        "text": "Ao",
        "start": 23,
        "end": 25
      }
      // ... 其他标题部分
    ]
  }
}
```

## 数据格式说明

训练数据来自 label studio 导出的 JSON 格式，每条数据包含：

- `text`: 原始文本
- `label`: 标注的实体列表，每个实体包含：
  - `start`: 起始位置
  - `end`: 结束位置
  - `text`: 实体文本
  - `labels`: 实体类型列表

## 模型说明

本项目使用基于 BERT 的命名实体识别模型，具体使用`bert-base-chinese`预训练模型，通过 SimpleTransformers 库进行微调。模型采用字符级别的标注，使用 BIO 标注方案（B-表示实体开始，I-表示实体内部，O 表示非实体）。

## 项目完成情况

- [x] 初始化项目，使用 pyvenv 管理 python 环境
- [x] 使用 SimpleTransformers 训练 NER 模型
- [x] 使用 SimpleTransformers 评估模型效果
- [x] 使用 SimpleTransformers 保存模型
- [x] 使用 SimpleTransformers 加载模型并进行推理
- [x] 使用 SimpleTransformers 保存推理结果

## 改进方向

1. 增加训练数据量，提高模型准确率
2. 优化模型参数，如学习率、批次大小等
3. 尝试不同的预训练模型，如 RoBERTa 等
4. 改进实体提取算法，合并相邻的同类型实体
5. 添加后处理规则，提高实体识别的准确性

