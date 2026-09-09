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
        query: A short, search-engine-style keyword query — NOT a full natural-language
        question. Strip filler words ("what is", "can you tell me", "I want to know"),
        pronouns, and politeness, and keep only the essential search terms, the way
        you'd type into a search engine.
        - Rewrite vague or conversational phrasing into concrete terms
          (e.g. convert "what's the weather like there today" into "weather Dhaka today").
        - Include a specific year, date, or "latest" when recency matters, since search
          results are ranked by relevance, not by your knowledge of the current date.
        - For follow-up questions referencing earlier context (pronouns like "it", "that",
          "the same thing"), resolve them into a standalone, self-contained query
          (e.g. convert "what about its release date?" into "iPhone 16 release date").
        - Keep it short — 3 to 8 words is usually ideal. Longer natural-language questions
          return noisier, less relevant results.

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

    """Retrieve relevant passages from the user's uploaded documents to answer
    questions about their specific content — facts, data, names, dates,
    numbers, or details that live in those files rather than in general
    knowledge. This is the only way to access document content; you cannot
    see uploaded files directly.

    WHEN TO CALL:
    Call this tool exactly once when the user's question could plausibly be
    answered by their uploaded documents — including questions about people,
    projects, figures, or specifics you would otherwise have to guess at.

    WHEN NOT TO CALL:
    - Greetings, small talk, or conversational filler
    - Math, logic, or calculations (use the calculator tool instead)
    - General knowledge you already know with confidence
    - Follow-up questions about tone, formatting, or phrasing rather than facts
    - Any question already answered earlier in this conversation

    CALL LIMIT — READ CAREFULLY:
    Call this tool at most once per user question. If the result indicates no
    relevant documents were found, that is a final result, not a signal to
    retry. Do not call this tool again with a rephrased, broadened, or
    alternate query in the same turn. Instead, immediately answer using your
    own general knowledge and clearly tell the user their documents did not
    contain the relevant information. Repeated calls for the same question
    waste time and essentially never surface something a well-formed first
    query missed.

    Args:
        query: A precise, standalone search query capturing exactly what the
            user wants to find — not the user's raw message. Resolve pronouns
            and vague references into concrete terms (e.g. "what about its
            pricing?" becomes "product pricing details"). Keep it focused on
            one specific piece of information; do not bundle multiple
            unrelated questions into one query.

    Returns:
        The most relevant retrieved passages with source attribution, or an
        explicit message stating no relevant documents were found — treat the
        latter as final, not as a prompt to try again.
    """


    vector_store = _get_vectorstore(state.get('kb_id', 'file_embeddings'))
    return generate_output(query, vector_store)