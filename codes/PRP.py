import os,json,re
from alive_progress import alive_bar

import torch

from hide_config import *

num_threads = '32'
torch.set_num_threads(int(num_threads))
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['OPENBLAS_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-mc","--model_checkpoint_path",type=str)  #本地：预训练模型名或本地目录；API：模型名称
#目前支持的选项（与下一条参数逐行对应）：
#GPT-3.5 ("gpt-3.5-turbo"和"gpt-3.5-turbo-16k"，根据输入文本长度自动判断)
#THUDM/chatglm3-6b或存储checkpoint的本地文件夹

parser.add_argument("-mn","--model_name",type=str)
#目前支持的选项
#TODO：OpenAI
#ChatGLM3

parser.add_argument("-dn","--dataset_name",type=str)
#目前支持的选项：
#Alg514

parser.add_argument("-ds","--dataset_path",type=str)
#Alg514数据：JSON文件对应的位置

parser.add_argument("-pt","--prompt_template",type=str,default="CoT")
#目前支持的选项：
#CoT: 在问题后面加“Let's think step by step.”
#pure：不加任何东西

parser.add_argument("-rt","--result_txt_path",type=str)  #输出结果储存文件（将直接追加写入）

args = parser.parse_args()
arg_dict=args.__dict__

