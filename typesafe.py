import os
from openrouter import OpenRouter
from chat_app_backend import AnswerCheck
import requests

client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

def _check_answer(question: str, answer: str) -> AnswerCheck:
    res = requests.post(
        "https://openrouter.ai/api/v1/system-one",  # verify exact path in OpenRouter's API reference
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
        json={
            "model": "typesafe/jev-1.13",
            "questions": {
                "is_relevant": {
                    "type": "noul",
                    "instructions": "Does the answer actually address the question, "
                                     "directly and accurately? Be strict but fair.",
                }
            },
            "state": f"Question: {question}\n\nAnswer: {answer}",
        },
        timeout=15,
    )
    res.raise_for_status()
    data = res.json()
    noul_value = data["nouls"]["is_relevant"]["noul"]
    return AnswerCheck(is_relevant=noul_value >= 0.5, reason=f"confidence: {noul_value:.2f}")


print(_check_answer('What is AI', 'Machine Learning'))