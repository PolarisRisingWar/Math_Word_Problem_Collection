import os, json, re, random
from alive_progress import alive_bar

import torch

from get_data import get_data
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

parser.add_argument(
    "-mc", "--model_checkpoint_path", type=str
)  # 本地：预训练模型名或本地目录；API：模型名称
# 目前支持的选项（与下一条参数逐行对应）：
# GPT-3.5 ("gpt-3.5-turbo"和"gpt-3.5-turbo-16k"，根据输入文本长度自动判断)
# THUDM/chatglm3-6b或存储checkpoint的本地文件夹

parser.add_argument("-mn", "--model_name", type=str)
# 目前支持的选项
# TODO：OpenAI
# ChatGLM3

parser.add_argument("-dn", "--dataset_name", type=str)
# 目前支持的选项：
# Alg514

parser.add_argument("-ds", "--dataset_path", type=str)
# Alg514数据：train/valid/test json储存的文件夹

parser.add_argument(
    "-rt", "--result_txt_path", type=str
)  # 输出结果储存文件（将直接追加写入）

# 以下默认参数都来自原论文
parser.add_argument("-K", "--max_iteration", type=int, default=5, help="最大迭代数")

# 以下默认参数都来自原论文官方代码
parser.add_argument("--max_length", type=int, default=256)

args = parser.parse_args()
arg_dict = args.__dict__

# 数据预处理
# answer是数值类对象
test_data = get_data(arg_dict["dataset_name"], arg_dict["dataset_path"])["test"]


# 构建模型
predict = get_infer(arg_dict["model_name"], arg_dict["model_checkpoint_path"])

# 运行模型
result_file = open(arg_dict["result_txt_path"], "a")

init_template = "Q: {question}. \nA: Let’s think step by step"
rectification_template = "Q: {question} (The answer is likely not {hypothesis}) . A: Let’s think step by step"
verification_template = "Q: {verify_problem} If we know the answer to the above question is {generated_answer}, what is the value of unknown variable X? (If X is irrelevant to the calculation process please answer 'Unknown')\nA: Let's think step by step."

amount_predict_right = 0


def get_verify_problem_and_answer(question, wrong_answer=[]):
    """mask problem中的数字为X"""
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", question)
    random.shuffle(numbers)
    for n in numbers:
        if float(n) not in wrong_answer:
            return question.replace(str(n), "X"), float(n)
    return "No Number", "No Number"


with alive_bar(len(test_data)) as bar:
    for i in range(len(test_data)):
        question = test_data[i]["question"]
        answer = test_data[i]["answer"]

        generated_values = []

        initial_answer = predict(
            init_template.format(question=question), max_length=arg_dict["max_length"]
        )

        print_result = "1. Initialization Reasoning Path: " + initial_answer

        initial_value = extract_number_from_prediction(
            predict, question, initial_answer, {"max_length": arg_dict["max_length"]}
        )

        print_result += (
            "\n2. Initialization Numerical Answer: "
            + str(initial_value)
            + "\n3. 迭代："
        )

        generated_values.append(initial_value)

        verify_problem, verify_answer = get_verify_problem_and_answer(question, [])

        incorrect_verify_answer = []
        incorrect_answer = []
        for iter_number in range(arg_dict["max_iteration"]):
            if verify_problem == "No Number":
                break
            verification_result = predict(
                verification_template.format(
                    verify_problem=verify_problem, generated_answer=generated_values[-1]
                ),
                max_length=arg_dict["max_length"],
            )
            while "Unknown" in verification_result:
                incorrect_verify_answer.append(verify_answer)
                verify_problem, verify_answer = get_verify_problem_and_answer(
                    question, incorrect_verify_answer
                )
                if verify_problem == "No Number":
                    break
                verification_result = predict(
                    verification_template.format(
                        verify_problem=verify_problem,
                        generated_answer=generated_values[-1],
                    ),
                    max_length=arg_dict["max_length"],
                )
            if verify_problem == "No Number":
                break
            print_result += (
                "\n第"
                + str(iter_number + 1)
                + "轮. X: "
                + str(verify_answer)
                + "\nVerified Reasoning Path: "
                + verification_result
            )

            verification_value = extract_number_from_prediction(
                predict,
                verify_problem,
                verification_result,
                {"max_length": arg_dict["max_length"]},
                "X",
            )
            print_result += "\n Verified Numerical Answer: " + str(verification_value)
            if abs(verification_value - verify_answer) < 0.00001:
                break

            incorrect_answer.append(generated_values[-1])
            hypothesis = "{" + ",".join([str(x) for x in incorrect_answer]) + "}"
            rectification_result = predict(
                rectification_template.format(question=question, hypothesis=hypothesis),
                max_length=arg_dict["max_length"],
            )
            print_result += "\nRectification Reasoning Path: " + rectification_result

            rectification_value = extract_number_from_prediction(
                predict,
                question,
                rectification_result,
                {"max_length": arg_dict["max_length"]},
            )
            print_result += "\nRectification Numerical Answer: " + str(
                rectification_value
            )

            if abs(rectification_value - generated_values[-1]) < 0.00001:
                break

            generated_values.append(rectification_value)

        if abs(generated_values[-1] - answer) < 0.00001:
            amount_predict_right += 1

        result_file.write(
            json.dumps(
                {
                    "model_prediction": print_result,
                    "predict_value": generated_values[-1],
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
