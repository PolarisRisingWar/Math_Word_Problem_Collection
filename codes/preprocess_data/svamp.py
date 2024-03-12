#https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json

import json, os, random

all_data = []
all_data_text = set()
dataset_name = "datasets/svamp"

with open(os.path.join(dataset_name, "SVAMP.json")) as f:
    data = json.load(f)
    for d in data:
        if not d["Body"].endswith("."):
            d["Body"] += ","
        all_data.append({"question": d["Body"]+" "+d["Question"], "answer": d["Answer"]})

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
