import os, json


def get_data(dataset_name: str, dataset_path: str):
    return_json = {}

    if dataset_name == "Alg514" or dataset_name == "AI2":

        for split_type in ["train", "valid", "test"]:
            if dataset_name == "Alg514":
                file_path = f"alg514_questions_1result_{split_type}.json"

            else:
                file_path = f"{split_type}.json"

            return_json[split_type] = [
                {"question": x["sQuestion"], "answer": float(x["lSolutions"][0])}
                for x in json.load(open(os.path.join(dataset_path, file_path)))
            ]
    elif dataset_name == "dolphin1878":
        for split_type in ["train", "valid", "test"]:
            file_path = f"{split_type}.json"
            return_json[split_type] = [
                {"question": x["text"], "answer": float(x["ans_simple"][0])}
                for x in json.load(open(os.path.join(dataset_path, file_path)))
            ]

    if dataset_name in ["Alg514", "dolphin1878"]:
        threshold = 0.001
    else:
        threshold = 0.00001
    return return_json, threshold
