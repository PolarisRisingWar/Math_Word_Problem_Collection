import os,json

def get_data(dataset_name:str,dataset_path:str):
    return_json={}
    if dataset_name=="Alg514":
        for split_type in ["train","valid","test"]:
            return_json[split_type]=[{"question":x["sQuestion"],"answer":x["lSolutions"][0]} for x in \
                json.load(open(os.path.join(dataset_path,
                                            f"alg514_questions_1result_{split_type}.json".format(split_type))))]
    return return_json