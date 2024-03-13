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
                {
                    "question": x["sQuestion"],
                    "answer": float(x["lSolutions"][0]),
                    "answer_with_reasoning": "The calculation formula for the question is:"
                    + " ".join(x["lEquations"])
                    + "#### "
                    + str(x["lSolutions"][0]),
                }
                for x in json.load(open(os.path.join(dataset_path, file_path)))
            ]
    elif dataset_name == "dolphin1878":
        for split_type in ["train", "valid", "test"]:
            file_path = f"{split_type}.json"
            return_json[split_type] = [
                {"question": x["text"], "answer": float(x["ans_simple"][0])}
                for x in json.load(open(os.path.join(dataset_path, file_path)))
            ]
    elif dataset_name in ["ASDiv", "Ape210K", "SVAMP"]:
        for split_type in ["train", "valid", "test"]:
            file_path = f"{split_type}.json"
            return_json[split_type] = [
                {"question": x["question"], "answer": x["answer"]}
                for x in json.load(open(os.path.join(dataset_path, file_path)))
            ]
    elif dataset_name == "Math23K":
        for split_type in ["train", "valid", "test"]:
            file_path = f"{split_type}23k_processed.json"
            return_json[split_type] = [
                {"question": x["original_text"], "answer": float(x["answer"])}
                for x in json.load(open(os.path.join(dataset_path, file_path)))
            ]
    elif dataset_name == "GSM8K":
        for split_type in ["train", "test"]:
            file_path = f"{split_type}.jsonl"
            this_list = []
            for line in open(os.path.join(dataset_path, file_path)):
                one_problem = json.loads(line)
                answer_str = one_problem["answer"]
                this_list.append(
                    {
                        "question": one_problem["question"],
                        "answer": float(
                            answer_str[answer_str.find("#### ") + 5 :].replace(",", "")
                        ),
                        "answer_with_reasoning": answer_str,
                    }
                )
            return_json[split_type] = this_list

    if dataset_name in ["Alg514", "dolphin1878", "Ape210K"]:
        threshold = 0.001
    else:
        threshold = 0.00001
    return return_json, threshold
