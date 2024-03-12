# https://github.com/chaochun/nlu-asdiv-dataset/blob/master/dataset/ASDiv.xml

import json, os, random
import xml.etree.ElementTree as ET

all_data = []
dataset_folder = "datasets/asdiv"
dataset_file = "ASDiv.xml"

# 解析XML文件
tree = ET.parse(os.path.join(dataset_folder, dataset_file))
root = tree.getroot()

# 遍历所有的Problem元素
for problem in root.find("ProblemSet").findall("Problem"):
    question = problem.find("Body").text + " " + problem.find("Question").text
    answer_text = problem.find("Answer").text
    try:
        answer_float = float(answer_text[: answer_text.find(" (")])
        all_data.append({"question": question, "answer": answer_float})
    except ValueError:
        pass


random.seed(20240308)

random.shuffle(all_data)

train_val_test_list = [8, 1, 2]
tvt_sum = sum(train_val_test_list)
tvt_ratio_list = [i / tvt_sum for i in train_val_test_list]
train_end_index = int(tvt_ratio_list[0] * len(all_data))
val_end_index = train_end_index + int(tvt_ratio_list[1] * len(all_data))

json.dump(
    all_data[:train_end_index],
    open(os.path.join(dataset_folder, "train.json"), "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    all_data[train_end_index:val_end_index],
    open(os.path.join(dataset_folder, "valid.json"), "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    all_data[val_end_index:],
    open(os.path.join(dataset_folder, "test.json"), "w"),
    ensure_ascii=False,
    indent=4,
)
