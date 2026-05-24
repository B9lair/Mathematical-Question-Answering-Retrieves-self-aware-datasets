# 数学问答检索自知数据集

面向数学问答模型**检索自知能力**训练的中文数据集——让模型在回答前自主判定是否需要检索外部知识。

---

## 概况

| 属性 | 说明 |
|------|------|
| 语言 | 中文 |
| 领域 | K-12 数学（七年级至高三） |
| 数据来源 | CMM-Math 公开试题集 + 人教版数学教材知识库 |
| 知识库规模 | 约 32,000 个文本块（定义、定理、典型例题） |
| 许可 | [MIT](LICENSE) |

包含两个 JSON 文件，分别服务于不同训练阶段：

### 1. SFT 数据——两轮对话格式

**文件：** `sft_data_v4_two_round.json`

| 统计 | 数值 |
|------|------|
| 总样本数 | 7,115 |
| `[Retrieve]` 标签 | 3,810（53.5%） |
| `[NoRetrieve]` 标签 | 3,305（46.5%） |
| 格式 | 两轮多轮对话 |

每条样本模拟一次两轮对话交互：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个数学问题解答助手。首先判断是否需要从外部知识库检索信息…"},
    {"role": "user", "content": "问题：2022 的绝对值是 ( ) …"},
    {"role": "assistant", "content": "[Retrieve]"},
    {"role": "user", "content": "检索结果：\n[文档1] 【概念】绝对值的定义…"},
    {"role": "assistant", "content": "逐一分析…\n\\boxed{A}"}
  ],
  "retrieve_label": "[Retrieve]",
  "level": "初中",
  "question": "2022 的绝对值是 ( ) …",
  "retrieved_documents": [
    {"score": 0.72, "content": "…", "title": "【概念】绝对值的定义", "source": "七年级上册…"}
  ],
  "ground_truth": "A"
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | list[dict] | OpenAI 格式多轮对话，共 5 轮（system → user → assistant 决策 → user 注入文档 → assistant 推理） |
| `retrieve_label` | str | 检索决策标签，`[Retrieve]` 或 `[NoRetrieve]` |
| `level` | str | 学段（初中/高中） |
| `question` | str | 原始数学题文本 |
| `retrieved_documents` | list[dict] | 知识库 Top5 检索结果，含相似度、标题、来源、全文 |
| `ground_truth` | str | 标准答案 |

训练时监督信号仅作用于最后一轮 assistant 输出（推理 + 答案）。

---

### 2. PPO 数据——平铺属性格式

**文件：** `分析题数据集_with_retrieval_v2.json`

| 统计 | 数值 |
|------|------|
| 总样本数 | 7,122 |
| `[Retrieve]` 标签 | 3,817（53.6%） |
| `[NoRetrieve]` 标签 | 3,305（46.4%） |
| 格式 | 平铺键值结构 |

```json
{
  "question": "2022 的绝对值是 ( ) …",
  "answer": "A",
  "ground_truth": "A",
  "source": "CMM-Math",
  "subject": "代数",
  "level": "初中",
  "question_type": "选择题",
  "retrieve_label": "[Retrieve]",
  "knowledge_points": "绝对值的定义",
  "retrieved_documents": [
    {"score": 0.72, "content": "…", "title": "【概念】绝对值的定义", "source": "七年级上册…"}
  ]
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | str | 数学题文本 |
| `answer` | str | 简短答案标签 |
| `ground_truth` | str | 完整标准答案 |
| `source` | str | 数据来源（CMM-Math） |
| `subject` | str | 学科分类 |
| `level` | str | 学段（小学/初中/高中） |
| `question_type` | str | 题型（选择题/填空题/解答题） |
| `retrieve_label` | str | 检索决策标签，由规则自动生成 |
| `knowledge_points` | str | 通过 DeepSeek API 提取的核心知识点 |
| `retrieved_documents` | list[dict] | 知识库 Top5 检索结果 |

`retrieve_label` 依据题目文本特征（是否含复杂符号、是否需多步推导、是否涉及外部概念）由规则自动生成，构建阶段经人工抽检核验。

---

## 知识库

知识库素材为人教版初高中全套数学教材内容，约 32,000 个文本块：

- 按章—节层级拆分，附语义类型标签：`[概念]`、`[原理]`、`[解题步骤]`、`[练习]`、`[例题]`
- 单块超过 500 字符做纵向切割，防止嵌入向量稀释
- 相似度阈值 ≥ 0.50（全部 32,102 条检索记录中，最低 0.500、最高 0.731、均值 0.677，约八成集中在 0.60–0.75）

检索前增设一道知识点提取前置步骤：调用 DeepSeek-Chat API，以题目文本为输入，输出核心知识点关键词，再将关键词提交至索引平台。

---

## 使用示例

### SFT 阶段

```python
import json
from transformers import AutoTokenizer

with open("sft_data_v4_two_round.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.add_special_tokens({"additional_special_tokens": ["[Retrieve]", "[NoRetrieve]"]})

for item in data:
    text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
    # tokenize 后送入模型训练
```

### PPO 阶段

```python
import json

with open("分析题数据集_with_retrieval_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    question = item["question"]
    docs = item["retrieved_documents"]
    gt = item["ground_truth"]
    label = item["retrieve_label"]
    # 送入模型生成并计算奖励
```

---

## 引用

使用本数据集请引用：

```bibtex
@misc{retrieval-awareness-dataset,
  title     = {Retrieval-Awareness Dataset for Mathematical QA},
  author    = {},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/mathrag-bench/retrieval-awareness-dataset}
}
```

---

## 相关链接

- **CMM-Math**：部分题目的来源公开试题集
- **Qwen2.5-7B-Instruct**：实验所用基座模型

如有问题请在 GitHub 提 issue。
