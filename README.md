# 代码库结构与文件说明

## 项目概述
该代码库主要围绕多个多模态大语言模型（Qwen2.5VL,Qwen2VL,KimiVL,kimi-audio等）的本地部署及应用展开，主要用于不同类型的情绪线索提取， personality 评分相关的任务处理，包含模型加载、提示词构建、训练数据处理等功能模块。

## 数据说明

### OPVA数据库标签信息说明

数据库中包含685位被试对8个问题（q1、q2、…、q8）的回答。评分者在被试回答每个问题后，对其相应的人格维度（factor）进行评分，分数取值范围为[1.0, 5.0]。

该数据库关注两个人格维度（factor）：

> * Extraversion（外向性）
> * Conscientiousness（尽责性）

其中，q1、q3、q5、q7问题用于激活Extraversion维度；q2、q4、q6、q8用于激活Conscientiousness维度。

---

### Prolific数据库标签信息说明

数据库中包含636位被试对6个问题（q1、q2、…、q6）的回答。评分者在被试回答每个问题后，对其相应的人格维度（factor）进行评分，分数取值范围为[1.0, 5.0]。

该数据库关注四个人格维度（factor）：

> * Honesty-Humility（诚实-谦逊）
> * Extraversion（外向性）
> * Agreeableness（宜人性）
> * Conscientiousness（尽责性）

其中，q1、q2为通用问题（General），不对应特定人格维度；q3问题用于激活Honesty-Humility维度；q4问题用于激活Extraversion维度；q5问题用于激活Agreeableness维度；q6问题用于激活Conscientiousness维度。

### 数据库主文件data.pkl
字典文件，包含了6个变量
| key                 | type      | shape/len  |
|:--------------------|-----------|------------|
| ground_truth_opva   | DataFrame | (685, 547) |
| ground_truth_prolific | DataFrame | (646, 35)  |
| meta_opva           | dict      | 7          |
| meta_prolific       | dict      | 6          |
| transcripts_opva    | DataFrame | (685, 9)   |
| transcripts_prolific| DataFrame | (636, 7)   |

#### ground_truth_opva
包含了所有685个被试的547个标签信息，其中重要的信息有：
```python
with open('data.pkl', 'rb') as file:
    ground_truth = pickle.load(file)
ground_truth = data["ground_truth_opva"]
d2 = ground_truth.columns
["Extraversion_observer_facet_mean","Conscientiousness_observer_facet_mean"]#他评人格分数，分别对应[Extraversion, Conscientiousness]
```

#### ground_truth_prolific
包含了所有685个被试的547个标签信息，其中重要的信息有：
```python
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)
ground_truth = data["ground_truth_prolific"]
d2 = ground_truth.columns

["H_observer", "A_observer", "C_observer", "E_observer"]  # 他评人格分数，对应[Honesty-Humility, Agreeableness, Conscientiousness, Extraversion]

```

####  meta_opva/meta_prolific 
包含了数据库的元信息，具体包括：

> - questions：字典文件，储存了每一个题目q_n对应的题目内容。
> - questions_key_personality：字典文件，存储了每一个题目q_n对应的人格维度。

####  transcripts_opva/transcripts_prolific 
> transcripts_opva: pandas dataframe 文件，包含了685个被试回答8个问题的文本内容。被试的编号储存在第一列中。
> transcripts_prolific: pandas dataframe 文件，包含了636个被试回答6个问题的文本内容。被试的编号储存在第一列中。



## 文件夹说明

### `prompt/`
 - 用于构建提示词（Prompt）相关的代码，包括构建情绪提取，人格评估提示词生成。

 ### `emotion/`
 - MLLM提取情绪的相关代码，包括从视频，音频等提取情绪。

### `get_answer/`
 - MLLM进行最终的人格评估的相关代码。


### `model.py`
```python
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/root/autodl-tmp')
```
 - 功能：从ModelScope下载预训练模型（如Qwen2.5-VL-7B-Instruct）到指定缓存目录，为后续模型加载提供本地路径。


