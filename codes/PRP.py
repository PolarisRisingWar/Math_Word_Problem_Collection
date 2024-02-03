import os,json
from alive_progress import alive_bar

import torch

from get_data import get_data
from prompts import question2prompt
from infer_predict import get_infer
from extract_number import extract_number_from_prediction

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
#Alg514数据：train/valid/test json储存的文件夹

parser.add_argument("-rt","--result_txt_path",type=str)  #输出结果储存文件（将直接追加写入）

#以下默认参数都来自原论文
parser.add_argument("-K","--max_iteration",type=int,default=5,help="最大迭代数")

args = parser.parse_args()
arg_dict=args.__dict__

#数据预处理
#answer是数值类对象
test_data=get_data(arg_dict["dataset_name"],arg_dict["dataset_path"])["test"]


#构建模型 
predict=get_infer(arg_dict["model_name"],arg_dict["model_checkpoint_path"])
        
#运行模型
result_file=open(arg_dict["result_txt_path"],"a")

amount_predict_right=0
with alive_bar(len(test_data)) as bar:
    for i in range(len(test_data)):
        model_prediction=predict(question2prompt(test_data[i]["question"]))

        predict_result=extract_number_from_prediction(predict,test_data[i]["question"],model_prediction)

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