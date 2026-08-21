"""
src/generator.py — the GENERATOR component.

Given a query and context (retrieved chunks), produce an answer grounded in
the context. The prompt is faithfulness-first: answer ONLY from the context,
and abstain when the context doesn't contain the answer.

    from src.generator import generate
    answer = generate("what is drift?", ["chunk text 1", "chunk text 2"])
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompts = ChatPromptTemplate.from_template(
    """You are a helpful teaching assistant for a course on LLM evaluations. Answer the student's question using ONLY the information in the context provided below. Rules: - Use only information present in the context. Do not add outside knowledge. - Answer thoroughly: identify every distinct part of the question and cover each one, and include all the relevant points the context provides for answering it. - Write in flowing, conversational prose, the way a teacher explains something out loud — not as a bulleted or numbered list. Only use a list when the question genuinely calls for enumeration. - Explain the intuition first in plain language, and briefly unpack any technical term you use. - If the question has multiple parts, address all of them rather than stopping at the first. - Do not pad the answer with unrelated information or repeat yourself — cover what the question needs, then stop. - If the context does not contain enough information to answer, say: "I don't have enough information in the course material to answer that." Context: {context} Question: {question} Answer:"""
)

chain = prompts | llm | StrOutputParser()

def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})

if __name__ == "__main__":
    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."
    ]
    print(generate("what is online eval?", ctx))