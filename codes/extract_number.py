import re
from typing import Union, Callable


def extract_result_prompt(question, nl_result, x="the answer") -> str:
    """参考Get an A in Math: Progressive Rectification Prompting"""
    return f"""
        Q: {question} 
        A: {nl_result} 
        Therefore, {x} (expressed in Arabic numerals and without units) is:
        """


def extract_last_number(text) -> float:
    """从自然语言格式的结果中抽取出最后一个数值"""
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if not numbers:
        return 0
    last_number = numbers[-1]
    if "." in last_number:
        return float(last_number)
    else:
        i = int(last_number)
        try:
            return float(i)
        except OverflowError:  # OverflowError: int too large to convert to float
            return 0


def extract_number_from_prediction(
    predict_function: Callable[[str], str],
    question: str,
    prediction: str,
    predict_function_params: dict = None,
    x: str = "the answer",
) -> Union[int, float]:
    if predict_function_params is not None:
        predict_result = predict_function(
            extract_result_prompt(question, prediction, x), **predict_function_params
        )
    else:
        predict_result = predict_function(
            extract_result_prompt(question, prediction, x)
        )
    try:
        predict_result = float(predict_result)
    except:
        predict_result = extract_last_number(predict_result)
        if predict_result is None:
            predict_result = 0

    return predict_result
