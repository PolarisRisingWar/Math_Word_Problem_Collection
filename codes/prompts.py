# prompt template


def question2prompt(question: str, template_name: str) -> str:
    if template_name == "CoT":
        return question + " Let's think step by step."
    elif template_name == "pure":
        return question
    elif template_name == "CoT+tip":
        return (
            question
            + " Let's think step by step. I will tip you $100,000 for a perfect answer."
        )
