from langchain_groq import ChatGroq
from groq import RateLimitError
from dotenv import load_dotenv

load_dotenv()

MODEL_CHAIN = [
    ChatGroq(model='openai/gpt-oss-120b', temperature=0.2),
    ChatGroq(model='openai/gpt-oss-20b', temperature=0.2),
    ChatGroq(model='qwen/qwen3.6-27b', temperature=0.2),
]


def invoke_with_fallback(prompt_or_messages, tools=None):
    """Invoke the model chain, falling back to the next model on RateLimitError."""
    last_error = None
    for model in MODEL_CHAIN:
        try:
            bound = model.bind_tools(tools) if tools else model
            return bound.invoke(prompt_or_messages)
        except RateLimitError as e:
            last_error = e
            continue
    raise last_error  # every model in the chain is rate-limited