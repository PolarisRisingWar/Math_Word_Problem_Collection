# https://www.microsoft.com/en-us/research/uploads/prod/2016/02//dolphin-number_word_std.zip
# 以下直接使用解压后的文件（其实我没太看懂原数据集的结构，以后我可能会看看原论文重新处理一下）

# Shuming Shi, Yuehui Wang, Chin-Yew Lin, Xiaojiang Liu, and Yong Rui. 2015. Automatically Solving Number Word Problems by Semantic Parsing and Reasoning. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1132–1142, Lisbon, Portugal. Association for Computational Linguistics.

# dolphin1878只分了dev和test，我直接重新划分数据集了
import json, os, random

all_data = []
all_data_text = set()
dataset_name = "datasets/dolphin1878"

for file in ["number_word_std.dev.json", "number_word_std.test.json"]:
    with open(os.path.join(dataset_name, file)) as f:
        data = json.load(f)
        for d in data:
            if (not d["text"] in all_data_text) and (len(d["ans_simple"]) == 1) and (not "or" in d["ans"]):
                all_data.append(d)
                all_data_text.add(d["text"])

random.seed(20240308)

random.shuffle(all_data)

train_val_test_list = [8, 1, 2]
tvt_sum = sum(train_val_test_list)
tvt_ratio_list = [i / tvt_sum for i in train_val_test_list]
train_end_index = int(tvt_ratio_list[0] * len(all_data))
val_end_index = train_end_index + int(tvt_ratio_list[1] * len(all_data))

json.dump(
    all_data[:train_end_index],
    open(os.path.join(dataset_name, "train.json"), "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    all_data[train_end_index:val_end_index],
    open(os.path.join(dataset_name, "valid.json"), "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    all_data[val_end_index:],
    open(os.path.join(dataset_name, "test.json"), "w"),
    ensure_ascii=False,
    indent=4,
)
