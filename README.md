数值推理，包括但不限于（具体的分类不一定是严格MECE的）：  
1. 数值推理
2. MWP
3. 信息抽取：数值抽取
4. 数值表征



超过5M的文件都存储在了百度网盘上，以方便大陆用户下载：  
链接：  
提取码：

* [1. 数据](#数据)
* [2. 论文](#论文)
* [3. 工具](#工具)

# 数据
简单介绍：
|**数据集名称**|**语言**|**下载和预处理策略**|**出处**|**任务**|
|---|---|---|-----|---|


其他相关数据集介绍：  


# 论文
**2022年**  
1. 数值推理
    1. (ACL) [Turning Tables: Generating Examples from Semi-structured Tables for Endowing Language Models with Reasoning Skills](https://aclanthology.org/2022.acl-long.416/)：表格数据
    2. (AAAI) [Weakly Supervised Neuro-Symbolic Module Networks for Numerical Reasoning](https://arxiv.org/abs/2101.11802)

**2021年**  
1. MWP
    1. (NAACL) [Are NLP Models really able to Solve Simple Math Word Problems?](https://arxiv.org/abs/2103.07191)
2. 数值表征
    1. (NAACL) [Representing Numbers in NLP: a Survey and a Vision](https://aclanthology.org/2021.naacl-main.53/)

**2020年**
1. 数值推理
    1. (EMNLP) [Question Directed Graph Attention Network for Numerical Reasoning over Text](https://aclanthology.org/2020.emnlp-main.549/)：改进NumNet，用异质有向图将类型（单位）和实体信息也结合进来，做数值推理
2. 数值常识
    1. (EMNLP) [Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-Trained Language Models](https://aclanthology.org/2020.emnlp-main.557/)：通过数值表征学习，LLM能获得数值所处的大致区间
2. MWP
    1. (EMNLP) [Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems](https://arxiv.org/abs/2010.06823)
3. 数值表征
    1. (EMNLP) [Learning Numeral Embeddings](https://arxiv.org/abs/2001.00003)
4. (ACL) [Injecting Numerical Reasoning Skills into Language Models](https://aclanthology.org/2020.acl-main.89/)：logarithmic difference能够给小数字更高权重

**2019年**  
1. 数值推理
    1. (EMNLP) [NumNet: Machine Reading Comprehension with Numerical Reasoning](https://aclanthology.org/D19-1251/)：comparison-aware GNN推理数值之间的相对比较关系
    2. (NAACL) [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://aclanthology.org/N19-1246/)：实现数值之间的计数、加减等操作
2. MWP
    1. (NAACL) [MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms](https://aclanthology.org/N19-1245/)

**2018年**  
1. (ACL) [Numeracy for Language Models: Evaluating and Improving their Ability to Predict Numbers](https://aclanthology.org/P18-1196/)：用MAPE (median absolute percentage error) 作为损失函数

**2016年** 
1. MWP
    1. (ACL) [How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation](https://aclanthology.org/P16-1084/)

**2011年**  
1. 数值抽取
    1. (WWW) [SCAD: collective discovery of attribute values](https://dl.acm.org/doi/abs/10.1145/1963405.1963469)

**2009年**  
1. (SIGIR) [Learning to rank for quantity consensus queries](https://dl.acm.org/doi/10.1145/1571941.1571985)：检索任务，根据数值排序

# 工具
1. 