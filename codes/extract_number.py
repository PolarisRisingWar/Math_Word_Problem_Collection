import re

def extract_result_prompt(question,nl_result):
    """参考Get an A in Math: Progressive Rectification Prompting"""
    return f"""
        Q: {question} 
        A: {nl_result} 
        Therefore, X (expressed in Arabic numerals and without units) is:
        """

def extract_last_number(text):
    """从自然语言格式的结果中抽取出最后一个数值"""
    # 使用正则表达式找到所有数字（整数或浮点数）
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    
    # 如果没有找到数字，返回None
    if not numbers:
        return None
    
    # 返回最后一个数字，转换为相应的整数或浮点数
    last_number = numbers[-1]
    if '.' in last_number:
        return float(last_number)
    else:
        return int(last_number)

def extract_number_from_prediction(predict_function:function,question:str,prediction:str):
    predict_result=predict_function(extract_result_prompt(question,prediction))
    try:
        predict_result=float(predict_result)
    except:
        predict_result=extract_last_number(predict_result)
        if predict_result is None:
            predict_result=0
    
    return predict_result