import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langsmith import Client
from pydantic import BaseModel, Field

from chat_app_backend_rag import (
    _get_vectorstore,
    _load_and_split,
    add_documents_to_store,
    clear_collection,
    generate_output,
)

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"

PROJECT_ROOT = Path(r"D:\AI-ML\AI Projects\fastapi-chat-app")
EVAL_DIR = PROJECT_ROOT / "eval_files"
PDF_PATHS = sorted(EVAL_DIR.glob("*.pdf"))
EVAL_COLLECTION = "langsmith_document_eval"
DATASET_NAME = "personal-ai-assistant-document-rag-v1"

if len(PDF_PATHS) < 2:
    raise FileNotFoundError(f"Expected at least two PDFs in {EVAL_DIR}, found: {PDF_PATHS}")

client = Client()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, max_tokens=512)

all_chunks = []
for pdf_path in PDF_PATHS:
    chunks = _load_and_split([str(pdf_path)])
    for chunk in chunks:
        chunk.metadata["source"] = pdf_path.name
    all_chunks.extend(chunks)

print(f"Loaded {len(all_chunks)} chunks from {len(PDF_PATHS)} PDFs")
print("Sources:")
for source in sorted({chunk.metadata.get("source") for chunk in all_chunks}):
    count = sum(chunk.metadata.get("source") == source for chunk in all_chunks)
    print(f"- {source}: {count} chunks")

try:
    clear_collection(EVAL_COLLECTION)
except Exception:
    pass
add_documents_to_store([str(path) for path in PDF_PATHS], collection_name=EVAL_COLLECTION)
print(f"Indexed documents into Chroma collection: {EVAL_COLLECTION}")


class GeneratedQA(BaseModel):
    question: str = Field(description="A specific question answerable from the supplied passage")
    answer: str = Field(description="A concise answer supported only by the supplied passage")
    evidence: str = Field(description="A short exact quote or faithful excerpt supporting the answer")


def make_candidate_examples(chunks_per_document: int = 5) -> list[dict]:
    candidates = []
    for source in sorted({chunk.metadata.get("source") for chunk in all_chunks}):
        source_chunks = [chunk for chunk in all_chunks if chunk.metadata.get("source") == source]
        selected = source_chunks[:chunks_per_document]
        for chunk in selected:
            prompt = PromptTemplate.from_template(
                """
                Create one evaluation question and answer from this document passage.
                The question must require information in the passage, not outside knowledge.
                The answer must be concise, accurate, and contain no unsupported claims.
                Include a short evidence excerpt from the passage.

                Source: {source}
                Passage:
                {passage}
                """
            ).format(source=source, passage=chunk.page_content)
            qa = llm.with_structured_output(GeneratedQA).invoke(prompt)
            candidates.append(
                {
                    "question": qa.question,
                    "answer": qa.answer,
                    "evidence": qa.evidence,
                    "source": source,
                    "page": chunk.metadata.get("page"),
                }
            )
    return candidates[:10]

candidate_examples = make_candidate_examples()
print(f"Generated {len(candidate_examples)} candidate examples")
for idx, example in enumerate(candidate_examples, 1):
    print(f"[{idx}] {example['question']}")
    print(f"A: {example['answer']}")
    print(f"SRC: {example['source']}")
    print("---")

# Review checkpoint: you can edit the list before evaluating.
reviewed_examples = candidate_examples
assert len(reviewed_examples) == 10
assert all(example["question"].strip() for example in reviewed_examples)
assert all(example["answer"].strip() for example in reviewed_examples)
print("Review checkpoint passed. Examples are ready for LangSmith.")

answer_prompt = PromptTemplate.from_template(
    """
    Answer the question using only the retrieved document context below.
    If the context does not answer the question, say that the documents do not contain enough information.
    Do not invent facts or citations. Keep the answer concise.

    Question: {question}
    Retrieved context, ordered from highest-ranked to lowest-ranked:
    {context}
    """
)


def evaluate_target(inputs: dict) -> dict:
    question = inputs["question"]
    started = time.perf_counter()
    context = asyncio.run(generate_output(question, _get_vectorstore(EVAL_COLLECTION)))
    response = llm.invoke(answer_prompt.format(question=question, context=context)).content
    return {
        "response": response,
        "context": context,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
if existing_datasets:
    dataset = existing_datasets[0]
    print(f"Reusing LangSmith dataset: {dataset.id}")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Reviewed document-grounded QA examples for the Personal AI Assistant RAG pipeline.",
    )
    client.create_examples(
        inputs=[{"question": example["question"]} for example in reviewed_examples],
        outputs=[
            {
                "answer": example["answer"],
                "evidence": example["evidence"],
                "source": example["source"],
                "page": example["page"],
            }
            for example in reviewed_examples
        ],
        dataset_id=dataset.id,
    )
    print(f"Created LangSmith dataset: {dataset.id}")


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str


def make_judge(key: str, instructions: str):
    judge_prompt = PromptTemplate.from_template(
        """
        You are evaluating a document-grounded AI assistant.
        Return a score from 0.0 to 1.0 and a short reason.

        Evaluation rule:
        {instructions}

        Question: {question}
        Reference answer: {reference_answer}
        Assistant answer: {answer}
        Retrieved context: {context}
        """
    )

    def evaluator(inputs, outputs, reference_outputs):
        result = llm.with_structured_output(JudgeScore).invoke(
            judge_prompt.format(
                instructions=instructions,
                question=inputs.get("question", ""),
                reference_answer=(reference_outputs or {}).get("answer", ""),
                answer=outputs.get("response", ""),
                context=outputs.get("context", ""),
            )
        )
        return {"key": key, "score": result.score, "comment": result.reason}

    evaluator.__name__ = key
    return evaluator


correctness = make_judge(
    "answer_correctness",
    "Score how accurately the assistant answer matches the reference answer. Give 1 only when the answer is materially correct; penalize wrong or missing facts.",
)
answer_relevance = make_judge(
    "answer_relevance",
    "Score how directly and usefully the assistant answer addresses the question. Penalize unrelated, evasive, or excessively vague answers.",
)
faithfulness = make_judge(
    "faithfulness",
    "Score whether every important claim in the assistant answer is supported by the retrieved context. Penalize unsupported claims and contradictions.",
)
completeness = make_judge(
    "completeness",
    "Score whether the assistant answer covers the important parts of the reference answer without omitting material details.",
)

print("Evaluators ready: correctness, answer_relevance, faithfulness, completeness")

result = client.evaluate(
    evaluate_target,
    data=DATASET_NAME,
    evaluators=[correctness, answer_relevance, faithfulness, completeness],
    experiment_prefix="document-rag-v1",
    max_concurrency=1,
)
print("LangSmith evaluation started.")
print(result)
