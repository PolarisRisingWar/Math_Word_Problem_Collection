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

parser.add_argument("-mc","--model_checkpoint_path",type=str)  #本地：预训练模型checkpoint目录；API：模型名称
#目前支持的选项：
#GPT-3.5 ("gpt-3.5-turbo"和"gpt-3.5-turbo-16k"，根据输入文本长度自动判断)

parser.add_argument("-dn","--dataset_name",type=str)
#目前支持的选项：
#Alg514

parser.add_argument("-ds","--dataset_path",type=str)
#Alg514数据：JSON文件对应的位置

parser.add_argument("-rt","--result_txt_path",type=str)  #输出结果储存文件（将直接追加写入）

args = parser.parse_args()
arg_dict=args.__dict__

#数据预处理
#answer是数值类对象
if arg_dict["dataset_name"]=="Alg514":
    test_data=[{"question":x["sQuestion"],"answer":x["lSolutions"][0]} for x in \
               json.load(open(arg_dict["dataset_path"]))]


#构建模型

def extract_result(text):
    """从自然语言格式的结果中抽取出结果，结果是最后一个数值"""
    # 使用正则表达式找到所有数字（整数或浮点数）
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    
    # 如果没有找到数字，返回None
    if not numbers:
        return None
    
    # 返回最后一个数字，转换为相应的整数或浮点数
    last_number = numbers[-1]
    if '.' in last_number:
        return float(last_number)
    else:
        return int(last_number)
    


if arg_dict["model_checkpoint_path"]=="GPT-3.5":
    import tiktoken
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random_exponential,
    )
    from openai import OpenAI

    client = OpenAI(api_key=CHATGPT3_5_API_KEY,base_url=CHATGPT3_5_base_url)

    def num_tokens_from_messages(messages,model):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for _, value in message.items():
                num_tokens += len(encoding.encode(value))
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    
    @retry(wait=wait_random_exponential(min=20, max=600),stop=stop_after_attempt(6))
    def predict(question):
        return wrap4predict(question)
    
    def wrap4predict(question):
        messages=[
                {"role": "user", "content": question+" Let's think step by step."}
            ]
        
        if num_tokens_from_messages(messages,"gpt-3.5-turbo")<3096:  #留1000个token给输出
            model_name="gpt-3.5-turbo"
        else:
            model_name="gpt-3.5-turbo-16k"

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return completion.choices[0].message.content
        
#运行模型
result_file=open(arg_dict["result_txt_path"],"a")

amount_predict_right=0
with alive_bar(len(test_data)) as bar:
    for i in range(len(test_data)):
        model_prediction=predict(test_data[i]["question"])
        predict_result=extract_result(model_prediction)
        if abs(predict_result-test_data[i]["answer"])<0.00001:
            amount_predict_right+=1
        
        result_file.write(json.dumps({"model_prediction":model_prediction,"predict_result":predict_result},
                                     ensure_ascii=False)+"\n")
        result_file.flush()

        bar()
        #break

result_file.close()

print(amount_predict_right)
print(amount_predict_right/len(test_data))