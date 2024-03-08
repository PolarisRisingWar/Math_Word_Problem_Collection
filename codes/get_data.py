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

    return return_json
