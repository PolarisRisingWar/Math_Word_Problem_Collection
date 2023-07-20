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
|AI2|英语|数据集下载地址挂了，但是可以在代码文件（<https://www.dropbox.com/s/1slbq2fi77fq7wx/Java%20code_mathproblems.zip?dl=1>）里面找到Math_Word_DS2.zip文件|(2014 EMNLP) [Re42：读论文 ARIS Learning to Solve Arithmetic Word Problems with Verb Categorization](https://blog.csdn.net/PolarisRisingWar/article/details/131726944)|MWP|395个样本|这个数据集名是MathDQN起的
|number_word_std / Dolphin / Dophin1878||<https://www.microsoft.com/en-us/research/uploads/prod/2016/02//dolphin-number_word_std.zip>|(2015 EMNLP) [Automatically Solving Number Word Problems by Semantic Parsing and Reasoning](https://aclanthology.org/D15-1135/)|MWP||
Dolphin18K||<https://www.microsoft.com/en-us/research/uploads/prod/2015/08/dolphin18k-v1.1.zip>|(2016 ACL) [How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation](https://aclanthology.org/P16-1084/)|MWP|18460个样本
DRAW-1K|英语|<https://www.microsoft.com/en-us/download/details.aspx?id=52628>|(2017 EACL) [Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems](https://aclanthology.org/E17-1047/)|MWP<br>（一元线性场景）|1000个样本
|Math23K|中文|<https://huggingface.co/datasets/Gxg/Math23K><br><https://github.com/SumbeeLei/Math_EN/tree/master/data>|(2017 EMNLP) [Deep Neural Solver for Math Word Problems](https://aclanthology.org/D17-1088/)|MWP<br>（一元线性场景）|23162个样本|腾讯人工智能实验室<br>数据来源于爬虫
|AQUA-RAT|英语|<https://github.com/deepmind/AQuA>|(2017 ACL) [Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems](https://arxiv.org/abs/1705.04146)|MWP|100000个样本
|GSM8K|英语|<https://huggingface.co/datasets/gsm8k>||MWP||介绍博文：[【搬运】GSM8K 数据集介绍_x66ccff的博客-CSDN博客](https://blog.csdn.net/qq_18846849/article/details/127547883)
|MATH|英语|<https://people.eecs.berkeley.edu/~hendrycks/MATH.tar>|(2021 NeurIPS) [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)|MWP
|Ape210K|中文|<https://github.com/Chenny0808/ape210k>||MWP


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
    1. (EMNLP) [Recall and Learn: A Memory-augmented Solver for Math Word Problems](https://aclanthology.org/2021.findings-emnlp.68)：REAL模型，类比/检索。REAL由存储模块、表示模块、类比模块、推理模块共4个模块组成，对于每个问题，首先通过存储模块检索类似的问题，然后用表示模块和类比模块对类似问题进行表征，二者都使用了自监督mask；最后用基于Copy机制的推理模块来实现公式生成
    官方GitHub项目：<https://github.com/sfeng-m/REAL4MWP>
    2. (EMNLP Findings) [Generate & Rank: A Multi-task Framework for Math Word Problems](https://aclanthology.org/2021.findings-emnlp.195/)：致力于解决用通用生成框架解决MWP的场景下的任务性细致优化：构建了一个多任务框架，基于生成式预训练语言模型（在论文中使用的是BART），同时学习生成（generate）和排序（rank），此外还设计了基于树的扰动和对排序器的在线更新机制。排序器是用实时更新的历史表达式数据库来训练的。
    2. [ ] (NeurIPS) [REAL2: An End-to-end Memory-augmented Solver for Math Word Problems](https://mathai4ed.github.io/papers/papers/paper_7.pdf)
    官方GitHub项目：<https://github.com/sfeng-m/REAL2>
    1. (NeurIPS) [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
    官方GitHub项目：<https://github.com/hendrycks/math/>
    5. (ACL) [Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?](https://aclanthology.org/2021.acl-long.75/)：提出NQG-T5模型，致力于解决seq2seq模型难以解决的域外compositional generalization问题，结合高精度的、基于语法的方法NQG和预训练seq2seq模型T5，在真实数据和标准评估数据上都表现良好。对于域内样本直接输出NQG，域外样本则输出T5结果。
    2. (NAACL) [Are NLP Models really able to Solve Simple Math Word Problems?](https://arxiv.org/abs/2103.07191)
2. 数值表征
    1. (NAACL) [Representing Numbers in NLP: a Survey and a Vision](https://aclanthology.org/2021.naacl-main.53/)

**2020年**
1. 数值推理
    1. (EMNLP) [Question Directed Graph Attention Network for Numerical Reasoning over Text](https://aclanthology.org/2020.emnlp-main.549/)：改进NumNet，用异质有向图将类型（单位）和实体信息也结合进来，做数值推理
2. 数值常识
    1. (EMNLP) [Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-Trained Language Models](https://aclanthology.org/2020.emnlp-main.557/)：通过数值表征学习，LLM能获得数值所处的大致区间
2. MWP
    1. (EMNLP) [Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems](https://arxiv.org/abs/2010.06823)
    2. (EMNLP) [A Knowledge-Aware Sequence-to-Tree Network for Math Word Problem Solving](https://aclanthology.org/2020.emnlp-main.579/)：KA-S2T模型，用基于树的表示学习方法，但结合了外部的常识性知识：用LSTM对问题进行嵌入，将问题中的实体和类型构建为实体图，用GAT结合外部知识实现表征，用tree-based decoder聚合state，以捕获长程依赖和全局表达式信息。
    官方GitHub项目：<https://github.com/qinzhuowu/KA-S2T>
    2. (COLING) [Solving Math Word Problems with Multi-Encoders and Multi-Decoders](https://aclanthology.org/2020.coling-main.262/)：用多种encoder和decoder来解决MWP任务：同时利用文本表征和将文本处理为依存句法树和数值比较信息的图后用图神经网络编码得到的表征，decoder也同时用基于序列和基于树的，最后会生成不同的公式，用这两个公式的损失函数合并为整个模型的优化目标。在推理时选择概率比较大的公式。
    4. (IEEE Transactions on Pattern Analysis and Machine Intelligence) [The Gap of Semantic Parsing: A Survey on Automatic Math Word Problem Solvers](https://arxiv.org/abs/1808.07290)：综述
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
    2. (NAACL) [Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems](https://aclanthology.org/N19-1272/)：将推理过程结合进了seq2seq模型之中：先用encoder表征问题中常数的语义信息（理解问题的语义），再用decoder依次决定公式中的数值和运算符，以模拟人类的推理逻辑。在decoder中，从原文中抽取或生成的数字组合成栈，逐一输出并生成匹配的运算符。最后生成的效果优于直接使用seq2seq模型

**2018年**  
1. (ACL) [Numeracy for Language Models: Evaluating and Improving their Ability to Predict Numbers](https://aclanthology.org/P18-1196/)：用MAPE (median absolute percentage error) 作为损失函数
2. MWP
    1. (EMNLP) [Translating a Math Word Problem to a Expression Tree](https://aclanthology.org/D18-1132/)：本文认为解决问题可以用多种形式的公式，这是seq2seq方法所无法解决的，因此将MWP问题映射为树（是对公式进行正则化）来建模。模型是encoder-decoder架构，将多种模型ensemble在一起，生成公式树的postorder traversal
    在Solving Math Word Problems with Multi-Encoders and Multi-Decoders中模型被称为Math-EN
    2. (AAAI) [MathDQN: Solving Arithmetic Word Problems via Deep Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11981)[^1]：强化学习。抽取数值，对数值两两配对、提取数值对的特征，结合上下文形成state，输入神经网络选择action，然后判断action选择正确与否（这个是选择运算符），正确reward为1，否则回去训练神经网络
    官方GitHub项目：[uestc-db/DQN_Word_Problem_Solver](https://github.com/uestc-db/DQN_Word_Problem_Solver)（Python 2的代码，不要啊，我不要复现这种东西）
    <font color='red'>（我自己不是搞强化学习的也没看论文所以妹整明白，就是它……运算符的选择空间不是挺小的吗？这事真的需要强化学习吗？）</font>
    3. (COLING) [Neural Math Word Problem Solver with Reinforcement Learning](https://aclanthology.org/C18-1018/)：CASS模型。强化学习。将复制与对齐机制结合进seq2seq模型，以解决生成数据不真实、错位的问题。强化学习训练框架使模型的训练损失函数与评估指标统一了。CASS也用模型输出作为分类模型的特征输入。

**2017年**  
1. MWP
    1. (EMNLP) [Re43：读论文 DNS Deep Neural Solver for Math Word Problems](https://blog.csdn.net/PolarisRisingWar/article/details/131772810)：第一篇用神经网络解决MWP问题的论文，直接将问题用RNN映射为公式。然后用结合RNN和基于相似度的检索模型，当检索模型相似度得分高于阈值时用检索结果的公式模版，反之用RNN。
    2. (ACL) [Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems](https://arxiv.org/abs/1705.04146)
    2. (EACL) [Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems](https://aclanthology.org/E17-1047/)

**2016年** 
1. MWP
    1. (ACL) [How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation](https://aclanthology.org/P16-1084/)：发现简单的基于相似度的方法就已经能超过大多数统计学习模型了

**2015年**  
1. MWP
    1. (EMNLP) [Automatically Solving Number Word Problems by Semantic Parsing and Reasoning](https://aclanthology.org/D15-1135/)
    2. (EMNLP) [Learn to Solve Algebra Word Problems Using Quadratic Programming](https://aclanthology.org/D15-1096/)：提出ZDC模型（KAZB模型的改进）
    3. (EMNLP) [Solving General Arithmetic Word Problems](https://arxiv.org/abs/1608.01413)
3. (TACL) [Reasoning about Quantities in Natural Language](https://aclanthology.org/Q15-1001/)：数值检测、Quantity Entailment和MWP任务

**2014年**  
1. MWP
    1. (EMNLP) [Re42：读论文 ARIS Learning to Solve Arithmetic Word Problems with Verb Categorization](https://blog.csdn.net/PolarisRisingWar/article/details/131726944)：第一篇非基于模板解决MWP的方法，解决加减算术问题。预测动词类型来进行题目分类，以及考虑其他一些人工抽取的特征，抽取题目中的实体、数值等信息，根据状态转移表得到公式
    2. (ACL) [Learning to Automatically Solve Algebra Word Problems](https://aclanthology.org/P14-1026/)：提出KAZB模型，是基于模版的方法：将问题映射为训练集中已有的公式模版

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

[^1]: [论文笔记 | MathDQN: Solving Arithmetric Word Problems via Deep Reinforcement Learning_ttliu_kiwi的博客-CSDN博客](https://blog.csdn.net/ting0922/article/details/104358369)
[【AAAI Oral】利用DeepMind的DQN解数学应用题，准确率提升15% - 知乎](https://zhuanlan.zhihu.com/p/33672372)
[mathdqn代码记录_ttliu_kiwi的博客-CSDN博客](https://blog.csdn.net/ting0922/article/details/104382898)