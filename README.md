数值推理，包括但不限于（具体的分类不一定是严格MECE的）：  
1. 数值推理
2. MWP
3. 信息抽取：数值抽取
4. 数值表征


* [1. 数据](#数据)
* [2. 论文](#论文)
* [3. 工具](#工具)

# 数据
简单介绍：
（由于数据可能太大所以我全都不上传Git了，但是反正全是放在datasets文件夹里，在代码中如有引用也全是从这里引用）
|**数据集名称**|**语言**|**下载地址**|**出处**|**任务**|**统计信息**|**其他**
|---|---|---|-----|---|----|--|--|
|Alg514|英语|<http://groups.csail.mit.edu/rbg/code/wordprobs/questions.json>|(2014 ACL) [Learning to Automatically Solve Algebra Word Problems](https://aclanthology.org/P14-1026/)|MWP<br>（线性场景）|514个样本
|number_word_std / Dolphin / Dophin1878||<https://www.microsoft.com/en-us/research/uploads/prod/2016/02//dolphin-number_word_std.zip>|(2015 EMNLP) [Automatically Solving Number Word Problems by Semantic Parsing and Reasoning](https://aclanthology.org/D15-1135/)|MWP||
Dolphin18K||<https://www.microsoft.com/en-us/research/uploads/prod/2015/08/dolphin18k-v1.1.zip>|(2016 ACL) [How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation](https://aclanthology.org/P16-1084/)|MWP|18K+样本
DRAW-1K|英语|<https://www.microsoft.com/en-us/download/details.aspx?id=52628>|(2017 EACL) [Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems](https://aclanthology.org/E17-1047/)|MWP<br>（一元线性场景）|1000个样本
|Math23K|中文|<https://huggingface.co/datasets/Gxg/Math23K>|(2017 EMNLP) [Deep Neural Solver for Math Word Problems](https://aclanthology.org/D17-1088/)|MWP<br>（一元线性场景）|23161个样本|腾讯人工智能实验室<br>数据来源于爬虫
|GSM8K|英语|<https://huggingface.co/datasets/gsm8k>||MWP||介绍博文：[【搬运】GSM8K 数据集介绍_x66ccff的博客-CSDN博客](https://blog.csdn.net/qq_18846849/article/details/127547883)



# 论文
**2023年**  
1. MWP
    1. (ACL) [Interpretable Math Word Problem Solution Generation Via Step-by-step Planning](https://arxiv.org/abs/2306.00784)：关注步骤分（bushi）
        1. [ ] 代码：GSM8K数据集
    2. (ACL) [Solving Math Word Problems via Cooperative Reasoning induced Language Models](https://arxiv.org/abs/2210.16257)
    2. (ACL Findings) [Compositional Mathematical Encoding for Math Word Problems](https://aclanthology.org/2023.findings-acl.635/)
    2. (BEA) [Scalable and Explainable Automated Scoring for Open-Ended Constructed Response Math Word Problems](https://aclanthology.org/2023.bea-1.12/)：关注MPT问题
    4. [Let GPT be a Math Tutor: Teaching Math Word Problem Solvers with Customized Exercise Generation](https://arxiv.org/abs/2305.14386)
    5. [Non-Autoregressive Math Word Problem Solver with Unified Tree Structure](https://arxiv.org/abs/2305.04556)
    6. [Solving Math Word Problems by Combining Language Models With Symbolic Solvers](https://arxiv.org/abs/2304.09102)


**2022年**  
1. 数值推理
    1. (ACL) [Turning Tables: Generating Examples from Semi-structured Tables for Endowing Language Models with Reasoning Skills](https://aclanthology.org/2022.acl-long.416/)：表格数据
    2. (AAAI) [Weakly Supervised Neuro-Symbolic Module Networks for Numerical Reasoning](https://arxiv.org/abs/2101.11802)
2. MWP
    1. (EMNLP) [Automatic Generation of Socratic Subquestions for Teaching Math Word Problems](https://arxiv.org/abs/2211.12835)
    2. (COLING) [WARM: A Weakly (+Semi) Supervised Model for Solving Math word Problems](https://arxiv.org/abs/2104.06722)

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
    1. (EMNLP) [NumNet: Machine Reading Comprehension with Numerical Reasoning](https://aclanthology.org/D19-1251/)：数值+GNN+数值之间的比较关系→在上下文中实现数值推理
    代码中文版：[j30206868/numnet-chinese: Modify numnet+ for chinese](https://github.com/j30206868/numnet-chinese)
    2. (NAACL) [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://aclanthology.org/N19-1246/)：实现数值之间的计数、加减等操作
2. MWP
    1. (NAACL) [MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms](https://aclanthology.org/N19-1245/)

**2018年**  
1. (ACL) [Numeracy for Language Models: Evaluating and Improving their Ability to Predict Numbers](https://aclanthology.org/P18-1196/)：用MAPE (median absolute percentage error) 作为损失函数

**2017年**  
1. MWP
    1. (EMNLP) [Deep Neural Solver for Math Word Problems](https://aclanthology.org/D17-1088/)：第一篇用神经网络解决MWP问题的方法
    2. (EACL) [Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems](https://aclanthology.org/E17-1047/)

**2016年** 
1. MWP
    1. (ACL) [How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation](https://aclanthology.org/P16-1084/)

**2015年**  
1. MWP
    1. (EMNLP) [Automatically Solving Number Word Problems by Semantic Parsing and Reasoning](https://aclanthology.org/D15-1135/)

**2014年**  
1. MWP
    1. (EMNLP) [Re42：读论文 ARIS Learning to Solve Arithmetic Word Problems with Verb Categorization](https://blog.csdn.net/PolarisRisingWar/article/details/131726944)：第一篇非基于模板解决MWP的方法，解决加减算术问题。预测动词类型来进行题目分类，以及考虑其他一些人工抽取的特征，抽取题目中的实体、数值等信息，根据状态转移表得到公式
    2. (ACL) [Learning to Automatically Solve Algebra Word Problems](https://aclanthology.org/P14-1026/)：基于模版的方法

**2011年**  
1. 数值抽取
    1. (WWW) [SCAD: collective discovery of attribute values](https://dl.acm.org/doi/abs/10.1145/1963405.1963469)

**2009年**  
1. (SIGIR) [Learning to rank for quantity consensus queries](https://dl.acm.org/doi/10.1145/1571941.1571985)：检索任务，根据数值排序

**1963年**  
1. MWP
    1. Computers and thought

# 工具
1. 