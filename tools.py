from langchain_community.tools import DuckDuckGoSearchRun
from chat_app_backend_rag import generate_output, _get_vectorstore
from langgraph.prebuilt import InjectedState
from typing import Annotated
from langchain.tools import tool
from sympy import sympify


_ddg_search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for current, real-time, or general-knowledge information
    NOT found in the user's uploaded documents.

    Use this tool when the user asks about:
    - Current events, news, or anything time-sensitive (prices, weather, scores, "latest", "today")
    - General facts not likely to be in their uploaded documents
    - Anything explicitly about the internet/web

    Do NOT use this tool for:
    - Questions about the user's own uploaded documents (use rag_tool instead)
    - Simple math (use calculator instead)
    - Greetings or small talk

    Args:
        query: A clear, standalone search query.

    Returns:
        A string summarizing the top search results.
    """
    try:
        results = _ddg_search.invoke(query)
        if not results:
            return "No relevant search results found."
        # print(results)

        return results
    except Exception as e:
        return f"Error performing web search: {e}"



@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = sympify(expression)
        if result.is_number:
            if result.is_integer:
                return str(int(result))
            result = result.evalf()
            # trim floating noise, keep a reasonable number of decimals
            result = round(float(result), 10)
        return str(result)
    except Exception as e:
        return f"Error: {e}"



@tool
def rag_tool(query: str, state: Annotated[dict, InjectedState]) -> str:

    """Search and retrieve relevant information from the user's uploaded documents 
    or knowledge base.

    Use this tool whenever the user asks a question that could be answered by 
    specific facts, data, definitions, or content that may exist in their documents 
    — including questions about people, projects, numbers, dates, or anything not 
    considered common/general knowledge.

    Do NOT use this tool for:
    - Greetings or small talk (e.g. "hi", "how are you")
    - Simple math or logic questions
    - General knowledge the model already knows confidently
    - Follow-up questions that are just clarifying tone/formatting, not facts

    Args:
        query: A clear, standalone search query representing what the user wants 
        to find. Rephrase vague or pronoun-heavy user questions into a specific, 
        self-contained query (e.g., convert "what about its pricing?" into 
        "product pricing details").

    Returns:
        A string containing the most relevant retrieved passages, or a message 
        indicating no relevant documents were found.
    """


    vector_store = _get_vectorstore(state.get('kb_id', 'file_embeddings'))
    return generate_output(query, vector_store)
