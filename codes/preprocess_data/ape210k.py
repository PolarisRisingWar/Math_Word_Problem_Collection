# https://github.com/Chenny0808/ape210k

import json, os, re

dataset_folder = "datasets/ape210k"


def replace_percentage_with_decimal(string):
    percentages = re.findall(r"(\d*\.?\d+)%", string)

    for percent in percentages:
        decimal_value = float(percent) / 100
        target = percent + "%"
        string = string.replace(target, str(decimal_value))

    return string


def add_prompt(x):
    data = json.loads(x)
    return_json = {}
    return_json["question"] = (
        "Please answer this elementary school math question. Please ensure that the obtained result is a numerical value. If the number is not divisible (whether it is a fraction or an irrational number), please retain 3 decimal places:"
        + " "
        + data["original_text"]
    )
    data["ans"] = replace_percentage_with_decimal(data["ans"])
    data["equation"] = replace_percentage_with_decimal(data["equation"])
    try:
        return_json["answer"] = float(data["ans"])
    except ValueError:
        try:
            if data["ans"].startswith("(") and data["ans"].endswith(")"):
                return_json["answer"] = eval(
                    data["ans"].replace("(", "").replace(")", "")
                )
            else:
                raise TypeError
        except TypeError:
            eq = data["equation"]
            try:
                return_json["answer"] = eval(eq[eq.find("=") + 1 :])
            except (TypeError, SyntaxError):
                pass  # 各种稀奇古怪的错误

    return return_json


for split_type in ["train", "valid", "test"]:
    file_path = f"{split_type}.ape.json"
    this_data = []
    for problem in open(os.path.join(dataset_folder, file_path)):
        processed_json = add_prompt(problem)
        if "answer" in processed_json:
            this_data.append(processed_json)

    json.dump(
        this_data,
        open(os.path.join(dataset_folder, split_type + ".json"), "w"),
        ensure_ascii=False,
        indent=4,
    )
