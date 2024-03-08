# 方法简介
1. LLM
    1. direct系列：直接将问题输入LLM
    另一种写法是在问题后面加`The answer is`
    GPT-3.5-Turbo&emsp;&emsp;ChatGLM3-6B
    2. CoT系列：在问题后面直接加` Let's think step by step.`
    另一种写法是：`Q: [Q]. A: Let’s think step by step`
    GPT-3.5-Turbo CoT
    3. PRP系列：算是CoT的一种延伸，通过重复多次CoT→验证答案→带着错误答案CoT的过程来实现预测

# 运行命令

GPT-3.5-Turbo + Alg514：`python codes/zero_shot_infer.py -mc GPT-3.5 -dn Alg514 -ds datasets/Alg514 -pt pure -rt codes/results/ChatGPT-3.5_Alg_result.txt`

GPT-3.5-Turbo + AI2：`python codes/zero_shot_infer.py -mc GPT-3.5 -dn AI2 -ds datasets/AI2 -pt pure -rt codes/results/ChatGPT-3.5_AI2_result.txt`（约4分钟）

GPT-3.5-Turbo CoT + Alg514：`python codes/zero_shot_infer.py -mc GPT-3.5 -dn Alg514 -ds datasets/Alg514 -pt CoT -rt codes/results/ChatGPT-3.5_COT_Alg_result.txt`

GPT-3.5-Turbo CoT+tip + Alg514：`python codes/zero_shot_infer.py -mc GPT-3.5 -dn Alg514 -ds datasets/Alg514 -pt CoT+tip -rt codes/results/ChatGPT-3.5_COT+tip_Alg_result.txt`

GPT-3.5_PRP + Alg514: `python codes/PRP.py -mc GPT-3.5 -dn Alg514 -ds datasets/Alg514 -rt codes/results/ChatGPT_PRP_Alg_result.txt`（约10分钟）

ChatGLM3-6B + Alg514：`CUDA_VISIBLE_DEVICES=4 python codes/zero_shot_infer.py -mc /data/pretrained_models/chatglm3-6b -mn ChatGLM3 -dn Alg514 -ds datasets/Alg514 -rt codes/results/ChatGLM3_Alg_result.txt`

GLM-4 + Alg514：`python codes/zero_shot_infer.py -mc GLM-4 -dn Alg514 -ds datasets/Alg514 -pt pure -rt codes/results/GLM4_Alg_result.txt`（约2.1元 + 6分钟）

