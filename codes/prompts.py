import re

#从自然语言格式的结果中抽取出结果

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

#prompt template

def question2prompt(question,template_name):
    if template_name=="CoT":
        return question+" Let's think step by step."
    elif template_name=="pure":
        return question
    elif template_name=="CoT+tip":
        return question+" Let's think step by step. I will tip you $100,000 for a perfect answer."