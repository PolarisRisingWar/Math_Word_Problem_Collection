MWP的综述也在路上啦，等我写完了跟导师说一声就去挂ArXiv上。如果跟我关系好而且等不及论文成品的可以私我加入共享石墨文档链接。

数值推理，包括但不限于（具体的分类不一定是严格MECE的）：  
1. 数值推理
2. MWP（标准做法依然是熟悉的文本生成，encoder+decoder。当然就是说虽然基础是这样的，但大家都可以积极整活）
3. 信息抽取：数值抽取
4. 数值表征

* [1. 实验结果](#实验结果)
* [2. 数据](#数据)
* [3. 论文](#论文)
* [4. 工具](#工具)

# 实验结果
MWP任务（setting见表后）的准确率指标：
| **方法名** | **Alg514** |
|---|---|
|GPT-3.5-Turbo|82.86%
|GPT-3.5-Turbo CoT|85.71%
|GPT-3.5-Turbo CoT+tip|80%
|GPT-3.5-Turbo CoT+SC|
|GPT-3.5-Turbo PRP|**94.29%**
|ChatGLM3-6B|65.71%
|GLM-4|77.14%
1. 仅考虑输出一个答案的数学题
2. 对于没有原始划分方案的数据集随机按照8:1:2的比例进行数据集划分：Alg514
3. tip的理论基础：[给ChatGPT小费真的好使！10块或10万效果拔群，但给1毛不升反降](https://mp.weixin.qq.com/s/vQPWFRMSrEzpsT-_N1VT3w)
4. SC (self-consistency)
5. PRP：(2024 AAAI) [Re61：读论文 PRP Get an A in Math: Progressive Rectification Prompting](https://blog.csdn.net/PolarisRisingWar/article/details/135844039)

# 数据
简单介绍：
（由于数据可能太大所以我全都不上传Git了，但是反正全是放在datasets文件夹里，在代码中如有引用也全是从这里引用）

尽量按时间顺序排列。有些我不确定先后顺序，所以可能有错误。

| **数据集名称** | **语言** | **下载地址** | **出处** | **任务** | **样本量** | **其他备注** |
|---|---|---|---|---|---|---|
| Alg514 | 英语 | <http://groups.csail.mit.edu/rbg/code/wordprobs/questions.json> | (2014 ACL) [Learning to Automatically Solve Algebra Word Problems](https://aclanthology.org/P14-1026/) | MWP<br>（线性场景） | 514 | |
| AI2 | 英语 | 数据集下载地址挂了，但是可以在代码文件（<https://www.dropbox.com/s/1slbq2fi77fq7wx/Java%20code_mathproblems.zip?dl=1>）里面找到Math_Word_DS2.zip文件 | (2014 EMNLP) [Re42：读论文 ARIS Learning to Solve Arithmetic Word Problems with Verb Categorization](https://blog.csdn.net/PolarisRisingWar/article/details/131726944) | MWP | 395 | 这个数据集名是MathDQN起的 |
| number_word_std / Dolphin / Dophin1878 | | <https://www.microsoft.com/en-us/research/uploads/prod/2016/02//dolphin-number_word_std.zip> | (2015 EMNLP) [Automatically Solving Number Word Problems by Semantic Parsing and Reasoning](https://aclanthology.org/D15-1135/) | MWP | | |
| Dolphin18K | | <https://www.microsoft.com/en-us/research/uploads/prod/2015/08/dolphin18k-v1.1.zip> | (2016 ACL) [How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation](https://aclanthology.org/P16-1084/) | MWP | 18460 | 公式+结果
| MAWPS | 英语 | [sroy9/mawps: Code for MAWPS: A Math Word Problem Repository](https://github.com/sroy9/mawps) | (2016 NAACL) [MAWPS: A Math Word Problem Repository](https://aclanthology.org/N16-1136/) | MWP |100K | |
| DRAW-1K | 英语 | <https://www.microsoft.com/en-us/download/details.aspx?id=52628> | (2017 EACL) [Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems](https://aclanthology.org/E17-1047/) | MWP<br>（一元线性场景） | 1000 | |
| Math23K | 中文 | <https://huggingface.co/datasets/Gxg/Math23K><br><https://github.com/SumbeeLei/Math_EN/tree/master/data> | (2017 EMNLP) [Deep Neural Solver for Math Word Problems](https://aclanthology.org/D17-1088/) | MWP<br>（一元线性场景） | 23162 | 腾讯人工智能实验室<br>数据来源于爬虫 |
| AQuA-RAT | 英语 | <https://github.com/deepmind/AQuA> | (2017 ACL) [Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems](https://arxiv.org/abs/1705.04146) | MWP | 100000个样本 | GSM8K嫌这个数据集里面模版化的题太多，而且自然语言解法的质量控制很拉 |
|MathQA|||(2019 NAACL) [MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms](https://aclanthology.org/N19-1245/)|MWP||AQUA-RAT的子集，关注解决AQuA-RAT中的错误。但是仍有约30%的数据存在不连续的问题
| DROP | | | (2019 NAACL) [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://aclanthology.org/N19-1246/) | 数值推理 | | |
| Academia Sinica Diverse MWP Dataset (ASDiv) V1.0 | 英语 | [chaochun/nlu-asdiv-dataset](https://github.com/chaochun/nlu-asdiv-dataset) | (2020 ACL) [A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers](https://aclanthology.org/2020.acl-main.92/) | MWP | 2.3K | 解决了之前数据集中的缺点 |
| Ape210K | 中文 | <https://github.com/Chenny0808/ape210k> | (2020) [Ape210K: A Large-Scale and Template-Rich Dataset of Math Word Problems](https://arxiv.org/pdf/2009.11506v1.pdf)（已撤回，所以在ArXiv论文主页是看不到的） | MWP | | 猿辅导 AI Lab，西北大学<br>包含 210K 个中国小学水平的数学问题，每个问题都包含黄金答案和得出答案所需的方程式 |
| MATH | 英语 | <https://people.eecs.berkeley.edu/~hendrycks/MATH.tar> | (2021 NeurIPS) [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874) | MWP | | GSM8K论文觉得这里面的问题有点太难了。问题来自可汗学院和Mathematica脚本 |
| GSM8K | 英语 | <https://huggingface.co/datasets/gsm8k> | (2021) [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) | MWP |7473条训练样本<br>1319条测试样本 | 众包生成 |
|Geometry3K|英语|<https://lupantech.github.io/inter-gps/>|(2021 ACL) [Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning](https://arxiv.org/abs/2105.04165)|图形QA|2401条训练样本<br>300条验证样本<br>601条测试样本
| SVAMP | 英语 | [arkilpatel/SVAMP: NAACL 2021: Are NLP Models really able to Solve Simple Math Word Problems?](https://github.com/arkilpatel/SVAMP) | (2021 NAACL) [Are NLP Models really able to Solve Simple Math Word Problems?](https://arxiv.org/abs/2103.07191) | MWP | | |
| 符号化的MWP | 英语 | [vedantgaur/Symbolic-MWP-Reasoning](https://github.com/vedantgaur/Symbolic-MWP-Reasoning) | (2023 ACL Findings) [Reasoning in Large Language Models Through Symbolic Math Word Problems](https://aclanthology.org/2023.findings-acl.364/) | MWP | | |
|SuperCLUE-Math6|中文||[SuperCLUE-Math6: 新一代中文数学推理数据集的探索之旅](https://mp.weixin.qq.com/s/jM2rgWE_-2TC7c49e22jAw)



# 论文
**2024年**  
1. MWP
    1. (AAAI) [Re61：读论文 PRP Get an A in Math: Progressive Rectification Prompting](https://blog.csdn.net/PolarisRisingWar/article/details/135844039)
    2. (清华) [Augmenting Math Word Problems via Iterative Question Composing](https://arxiv.org/abs/2401.09003)
    3. [Scaling the Authoring of AutoTutors with Large Language Models](https://arxiv.org/abs/2402.09216)
2. 几何题
    1. [GAPS: Geometry-Aware Problem Solver](https://arxiv.org/abs/2401.16287)
2. (ICDE) [Enhancing Quantitative Reasoning Skills of Large Language Models through Dimension Perception](https://arxiv.org/abs/2312.17532)：关注数值单位（维度）
3. [BIBench: Benchmarking Data Analysis Knowledge of Large Language Models](https://arxiv.org/abs/2401.02982)：这篇是商务智能那边数据分析领域的研究……也算是数值推理吧
4. [SuperCLUE-Math6: Graded Multi-Step Math Reasoning Benchmark for LLMs in Chinese](https://arxiv.org/abs/2401.11819)

**2023年**  
1. 数值推理
    1. (ACL) [A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models](https://arxiv.org/abs/2210.12023)
    3. (ICLR) [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610)
    2. (KDD) [Exploiting Relation-aware Attribute Representation Learning in Knowledge Graph Embedding for Numerical Reasoning](https://dl.acm.org/doi/abs/10.1145/3580305.3599338)
    3. (AAAI) [An Independent Evaluation of ChatGPT on Mathematical Word Problems (MWP)](https://arxiv.org/abs/2302.13814)
    4. (EMNLP) [MarkQA: A large scale KBQA dataset with numerical reasoning](https://aclanthology.org/2023.emnlp-main.633)
    5. (EMNLP) ATHENA: Mathematical Reasoning with Thought Expansion
    6. (EMNLP) UniMath: A Foundational and Multimodal Mathematical Reasoner
    7. (ICML) Large Language Models Can Be Easily Distracted by Irrelevant Context
    3. (EACL) [ComSearch: Equation Searching with Combinatorial Strategy for Solving Math Word Problems with Weak Supervision](https://arxiv.org/abs/2210.07017)
    4. (TMLR) [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://arxiv.org/abs/2211.12588)
    2. [Scaling Relationship on Learning Mathematical Reasoning with Large Language Models](https://arxiv.org/abs/2308.01825)：发现微调数据量越大，模型效果越好。提出RFT技术自动采样数据
    2. [CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models](https://blender.cs.illinois.edu/paper/creator2023.pdf)
    4. (港中文+腾讯) [StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving](https://arxiv.org/abs/2311.08803)：这一篇应该算是通用的解决方案，但是下游任务中包含数值推理
    5. (耶鲁等多家高校) [DocMath-Eval: Evaluating Numerical Reasoning Capabilities of LLMs in Understanding Long Documents with Tabular Data](https://arxiv.org/abs/2311.09805)：这一篇主要考虑带表格长文档的数值推理场景
    6. (人大、科大讯飞等) [Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning](https://arxiv.org/abs/2306.02408)
    7. [Learning From Mistakes Makes LLM Better Reasoner](https://arxiv.org/abs/2310.20689)
2. MWP
    1. (ACL OpenAI) [Interpretable Math Word Problem Solution Generation Via Step-by-step Planning](https://arxiv.org/abs/2306.00784)：关注步骤分（bushi）
        1. [ ] 代码：GSM8K数据集
    2. (ACL) [Solving Math Word Problems via Cooperative Reasoning induced Language Models](https://arxiv.org/abs/2210.16257)
    2. (ACL Findings) [Compositional Mathematical Encoding for Math Word Problems](https://aclanthology.org/2023.findings-acl.635/)
    4. (ACL Industry) [MathPrompter: Mathematical Reasoning using Large Language Models](https://arxiv.org/abs/2303.05398)
    4. (AAAI) [Generalizing Math Word Problem Solvers via Solution Diversification](https://arxiv.org/abs/2212.00833)
    5. (EMNLP) [Non-Autoregressive Math Word Problem Solver with Unified Tree Structure](https://arxiv.org/abs/2305.04556)
    4. (EMNLP) [Let GPT be a Math Tutor: Teaching Math Word Problem Solvers with Customized Exercise Generation](https://arxiv.org/abs/2305.14386)
    5. (EMNLP Findings) Conic10K: A Challenging Math Problem Understanding and Reasoning Dataset
    8. (TKDD) [Math Word Problem Generation via Disentangled Memory Retrieval](https://dl.acm.org/doi/10.1145/3639569)
    4. (IJCNN) [Improving Math Word Problems Solver with Logical Semantic Similarity](https://ieeexplore.ieee.org/abstract/document/10191088)
    5. (IJCNN) [Solving Math Word Problems Following Logically Consistent Template](https://ieeexplore.ieee.org/abstract/document/10191776)
    6. (NLPCC) [Solving Math Word Problem with Problem Type Classification](https://arxiv.org/abs/2308.13844)
    7. (ICANN) [Solving Math Word Problem with External Knowledge and Entailment Loss](https://link.springer.com/chapter/10.1007/978-3-031-44201-8_27)
    8. (IEEE International Conference on Big Data) [Combining Transformers and Tree-based Decoders for Solving Math Word Problems](https://www.computer.org/csdl/proceedings-article/bigdata/2023/10386340/1TUOgfb1Pe8)
    2. (BEA) [Scalable and Explainable Automated Scoring for Open-Ended Constructed Response Math Word Problems](https://aclanthology.org/2023.bea-1.12/)：关注MPT问题
    3. (ICLP) [Enhancing Math Word Problem Solving Through Salient Clue Prioritization: A Joint Token-Phrase-Level Feature Integration Approach](https://ieeexplore.ieee.org/abstract/document/10337252)
    4. (Computación y Sistemas) [Math Word Problem Solving: Operator and Template Techniques with Multi-Head Attention](https://www.polibits.cidetec.ipn.mx/ojs/index.php/CyS/article/view/4769)
    6. [Solving Math Word Problems by Combining Language Models With Symbolic Solvers](https://arxiv.org/abs/2304.09102)
    7. [Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification](https://arxiv.org/abs/2308.07921)：大意是用GPT-4 code interpreter，结合代码与文本生成更强的MWP结果（具体的还没看）
    8. [Exploring Equation as a Better Intermediate Meaning Representation for Numerical Reasoning](https://arxiv.org/abs/2308.10585)：通过方程而不是程序作为模型的中间输出（IMR），生成方程是通过LLM实现的
    9. (清华+智谱AI) [GPT Can Solve Mathematical Problems Without a Calculator](https://arxiv.org/abs/2309.03241)：提出MathGLM（GLM-10B改）
    10. [Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification](https://arxiv.org/abs/2308.07921)
    10. [Chatbots put to the test in math and logic problems: A preliminary comparison and assessment of ChatGPT-3.5, ChatGPT-4, and Google Bard](https://arxiv.org/abs/2305.18618)
    11. [An Empirical Study on Challenging Math Problem Solving with GPT-4](https://arxiv.org/abs/2306.01337)
    12. (耶鲁&卡梅) [ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics](https://arxiv.org/abs/2302.12433)：证明题
    13. [TinyGSM: achieving >80% on GSM8k with small language models](https://arxiv.org/abs/2312.09241)
3. 数值表征
    1. (TMLR) [Semantic Representations of Mathematical Expressions in a Continuous Vector Space](https://arxiv.org/abs/2211.08142)：表征数学表达式
4. 集合推理
    1. (谷歌) [GeomVerse: A Systematic Evaluation of Large Models for Geometric Reasoning](https://arxiv.org/abs/2312.12241)
3. 金融
    1. [Numerical Reasoning for Financial Reports](https://arxiv.org/abs/2312.14870)
4. (Nature DeepMind) [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6)：FunSearch模型，用函数搜索的方式解决数学问题
8. (ACL) [A Survey of Deep Learning for Mathematical Reasoning](https://aclanthology.org/2023.acl-long.817/)
3. (ACL Findings) [World Models for Math Story Problems](https://arxiv.org/abs/2306.04347)
4. (EMNLP) [MAF: Multi-Aspect Feedback for Improving Reasoning in Large Language Models](https://arxiv.org/abs/2310.12426)
9. (EMNLP Findings) [Large Language Models are Better Reasoners with Self-Verification](https://arxiv.org/abs/2212.09561)：有数值推理相关的下游任务
4. (EACL) [BERT is not The Count: Learning to Match Mathematical Statements with Proofs](https://arxiv.org/abs/2302.09350)
5. (华师) [Math-KG: Construction and Applications of Mathematical Knowledge Graph](https://arxiv.org/abs/2205.03772)
6. [Mathematical Language Models: A Survey](https://arxiv.org/abs/2312.07622)
7. [Bridging the Semantic-Numerical Gap: A Numerical Reasoning Method of Cross-modal Knowledge Graph for Material Property Prediction](https://arxiv.org/abs/2312.09744)

**2022年**  
1. 数值推理
    1. (ACL) [Turning Tables: Generating Examples from Semi-structured Tables for Endowing Language Models with Reasoning Skills](https://aclanthology.org/2022.acl-long.416/)：表格数据
    2. (AAAI) [Weakly Supervised Neuro-Symbolic Module Networks for Numerical Reasoning](https://arxiv.org/abs/2101.11802)
    3. (NeurIPS) [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
2. MWP
    1. (EMNLP) [Automatic Generation of Socratic Subquestions for Teaching Math Word Problems](https://arxiv.org/abs/2211.12835)
    2. (COLING) [WARM: A Weakly (+Semi) Supervised Model for Solving Math word Problems](https://arxiv.org/abs/2104.06722)
    3. (NAACL) [Practice Makes a Solver Perfect: Data Augmentation for Math Word Problem Solvers](https://aclanthology.org/2022.naacl-main.310/)
    4. (NAACL) [MWP-BERT: Numeracy-Augmented Pre-training for Math Word Problem Solving](https://arxiv.org/abs/2107.13435)

**2021年**  
1. MWP
    1. (EMNLP) [Recall and Learn: A Memory-augmented Solver for Math Word Problems](https://aclanthology.org/2021.findings-emnlp.68)：REAL模型，类比/检索。REAL由存储模块、表示模块、类比模块、推理模块共4个模块组成，对于每个问题，首先通过存储模块检索类似的问题，然后用表示模块和类比模块对类似问题进行表征，二者都使用了自监督mask；最后用基于Copy机制的推理模块来实现公式生成
    官方GitHub项目：<https://github.com/sfeng-m/REAL4MWP>
    2. (EMNLP) [Tree-structured Decoding for Solving Math Word Problems](https://aclanthology.org/D19-1241/)：用树形解码自上而下生成数学方程的抽象语法树。在解码过程中可以自动停止，不需要停止标记。
    3. (EMNLP) [Graph-to-Tree Neural Networks for Learning Structured Input-Output Translation with Applications to Semantic Parsing and Math Word Problem](https://aclanthology.org/2020.findings-emnlp.255/)
    2. (EMNLP Findings) [Generate & Rank: A Multi-task Framework for Math Word Problems](https://aclanthology.org/2021.findings-emnlp.195/)：致力于解决用通用生成框架解决MWP的场景下的任务性细致优化：构建了一个多任务框架，基于生成式预训练语言模型（在论文中使用的是BART），同时学习生成（generate）和排序（rank），此外还设计了基于树的扰动和对排序器的在线更新机制。排序器是用实时更新的历史表达式数据库来训练的。
    2. [ ] (NeurIPS) [REAL2: An End-to-end Memory-augmented Solver for Math Word Problems](https://mathai4ed.github.io/papers/papers/paper_7.pdf)
    REAL模型的进化版。
    官方GitHub项目：<https://github.com/sfeng-m/REAL2>
    1. (NeurIPS) [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
    官方GitHub项目：<https://github.com/hendrycks/math/>
    5. (ACL) [Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?](https://aclanthology.org/2021.acl-long.75/)：提出NQG-T5模型，致力于解决seq2seq模型难以解决的域外compositional generalization问题，结合高精度的、基于语法的方法NQG和预训练seq2seq模型T5，在真实数据和标准评估数据上都表现良好。对于域内样本直接输出NQG，域外样本则输出T5结果。
    6. (ACL | IJCNLP) [Measuring and Improving BERT’s Mathematical Abilities by Predicting the Order of Reasoning](https://aclanthology.org/2021.acl-short.49/)：用特殊的训练过程
    2. (NAACL) [Are NLP Models really able to Solve Simple Math Word Problems?](https://arxiv.org/abs/2103.07191)
    3. [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
    4. [Pretrained Language Models are Symbolic Mathematics Solvers too!](https://arxiv.org/abs/2110.03501)
2. 数值表征
    1. (NAACL) [Representing Numbers in NLP: a Survey and a Vision](https://aclanthology.org/2021.naacl-main.53/)
3. 数值推理
    1. [MathBERT: A Pre-Trained Model for Mathematical Formula Understanding](https://arxiv.org/abs/2105.00377)：第一个用于理解数学公式的预训练模型。预训练任务是预测从操作符树（OPT，公式的语义结构表示）中提取的掩码公式子结构，下游任务是数学信息检索、公式主题分类和公式标题生成
    2. (NeurIPS MATHAI4ED Workshop) [MathBERT: A Pre-trained Language Model for General NLP Tasks in Mathematics Education](https://arxiv.org/abs/2106.07340)：这个应该也算数值推理吧
    预训练数据集：从学前到大学
    任务：知识组件预测、自动分级开放式问答和知识追踪
    除此之外，本文还构建了数学领域的专属词典mathVocab

**2020年**
1. 数值推理
    1. (EMNLP) [Question Directed Graph Attention Network for Numerical Reasoning over Text](https://aclanthology.org/2020.emnlp-main.549/)：改进NumNet，用异质有向图将类型（单位）和实体信息也结合进来，做数值推理
2. 数值常识
    1. (EMNLP) [Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-Trained Language Models](https://aclanthology.org/2020.emnlp-main.557/)：通过数值表征学习，LLM能获得数值所处的大致区间
2. MWP
    1. (EMNLP) [Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems](https://arxiv.org/abs/2010.06823)
    2. (EMNLP) [A Knowledge-Aware Sequence-to-Tree Network for Math Word Problem Solving](https://aclanthology.org/2020.emnlp-main.579/)：KA-S2T模型，用基于树的表示学习方法，但结合了外部的常识性知识：用LSTM对问题进行嵌入，将问题中的实体和类型构建为实体图，用GAT结合外部知识实现表征，用tree-based decoder聚合state，以捕获长程依赖和全局表达式信息。
    官方GitHub项目：<https://github.com/qinzhuowu/KA-S2T>
    3. (EMNLP) [Point to the Expression: Solving Algebraic Word Problems using the Expression-Pointer Transformer Model](https://aclanthology.org/2020.emnlp-main.308/)
    3. (ICLR) [Deep Learning For Symbolic Mathematics](https://openreview.net/forum?id=S1eZYeHFDS)：这个解决的是符号积分和微分方程方面。这个算不算MWP我都不知道，反正先放到这里吧。
    4. (ICML) [Mapping Natural-language Problems to Formal-language Solutions Using Structured Neural Representations](https://dl.acm.org/doi/abs/10.5555/3524938.3525084)
    2. (COLING) [Solving Math Word Problems with Multi-Encoders and Multi-Decoders](https://aclanthology.org/2020.coling-main.262/)：用多种encoder和decoder来解决MWP任务：同时利用文本表征和将文本处理为依存句法树和数值比较信息的图后用图神经网络编码得到的表征，decoder也同时用基于序列和基于树的，最后会生成不同的公式，用这两个公式的损失函数合并为整个模型的优化目标。在推理时选择概率比较大的公式。
    4. (IEEE Transactions on Pattern Analysis and Machine Intelligence) [The Gap of Semantic Parsing: A Survey on Automatic Math Word Problem Solvers](https://arxiv.org/abs/1808.07290)：综述
3. 数值表征
    1. (EMNLP) [Learning Numeral Embeddings](https://arxiv.org/abs/2001.00003)
    2. (ICLR) [Neural Symbolic Reader: Scalable Integration of Distributed and Symbolic Representations for Reading Comprehension](https://openreview.net/forum?id=ryxjnREFwH)：符号表征也算数值表征吧
4. (ACL) [Injecting Numerical Reasoning Skills into Language Models](https://aclanthology.org/2020.acl-main.89/)：logarithmic difference能够给小数字更高权重
5. (KDD) [Self-Supervised Pretraining of Graph Neural Network for the Retrieval of Related Mathematical Expressions in Scientific Articles](https://arxiv.org/abs/2209.00446)：检索相似公式→检索相似论文

**2019年**  
1. 数值推理
    1. (NAACL) [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://aclanthology.org/N19-1246/)：实现数值之间的计数、加减等操作
    3. (EMNLP) [NumNet: Machine Reading Comprehension with Numerical Reasoning](https://aclanthology.org/D19-1251/)：数值+GNN+数值之间的比较关系→在上下文中实现数值推理
    代码中文版：[j30206868/numnet-chinese: Modify numnet+ for chinese](https://github.com/j30206868/numnet-chinese)
    3. (EMNLP | IJCNLP) [Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension](https://aclanthology.org/D19-1609/)：用BERT选择可执行程序（预定义好的）
2. MWP
    1. (IJCAI) [A Goal-Driven Tree-Structured Neural Model for Math Word Problems](https://www.ijcai.org/proceedings/2019/0736.pdf)
    2. (AAAI) [Template-Based Math Word Problem Solvers with Recursive Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/4697)
    2. (NAACL) [MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms](https://aclanthology.org/N19-1245/)
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
    2. (ACL) [Learning To Use Formulas To Solve Simple Arithmetic Problems](https://aclanthology.org/P16-1202/)

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