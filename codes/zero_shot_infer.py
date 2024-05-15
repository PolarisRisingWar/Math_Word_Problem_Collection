import os, json, time
from alive_progress import alive_bar

import torch

from get_data import get_data
from prompts import question2prompt
from infer_predict import get_infer
from extract_number import extract_number_from_prediction

num_threads = "32"
torch.set_num_threads(int(num_threads))
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import argparse

parser = argparse.ArgumentParser()

# 具体支持的选项可参考 codes/README.md

parser.add_argument(
    "-mc", "--model_checkpoint_path", type=str
)  # 本地：预训练模型名或本地目录；API：模型名称

parser.add_argument("-mn", "--model_name", type=str)
# 本地：模型名称

parser.add_argument("-dn", "--dataset_name", type=str)

parser.add_argument("-ds", "--dataset_path", type=str)
# train/valid/test json储存的文件夹

parser.add_argument("-pt", "--prompt_template", type=str, default="CoT")
# 目前支持的选项：
# CoT: 在问题后面加“Let's think step by step.”
# pure：不加任何东西
# CoT+tip: 在问题后面加“Let's think step by step. I will tip you $100,000 for a perfect answer.”

parser.add_argument(
    "-rt", "--result_txt_path", type=str
)  # 输出结果储存文件（将直接追加写入）

args = parser.parse_args()
arg_dict = args.__dict__

# 数据预处理
all_data, threshold = get_data(arg_dict["dataset_name"], arg_dict["dataset_path"])
test_data = all_data["test"]


# 构建模型
predict = get_infer(arg_dict["model_name"], arg_dict["model_checkpoint_path"])

# 运行模型
result_file = open(arg_dict["result_txt_path"], "a")

amount_predict_right = 0
with alive_bar(len(test_data)) as bar:
    for i in range(len(test_data)):
        if arg_dict["model_checkpoint_path"] == "moonshot-v1-8k":  # 说是RPM=3，但是我sleep 20秒也会超，不理解了真是
            time.sleep(60)
        model_prediction = predict(
            question2prompt(test_data[i]["question"], arg_dict["prompt_template"])
        )

        predict_value = extract_number_from_prediction(
            predict, test_data[i]["question"], model_prediction
        )

        if abs(predict_value - test_data[i]["answer"]) < threshold:
            amount_predict_right += 1

        result_file.write(
            json.dumps(
                {
                    "model_prediction": model_prediction,
                    "predict_value": predict_value,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        result_file.flush()

        bar()
        # break

result_file.close()

print(amount_predict_right)
print(amount_predict_right / len(test_data))
