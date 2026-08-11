# import os, re, glob, json, random
# from dotenv import load_dotenv
# from deepeval.synthesizer import Synthesizer
# from langchain_text_splitters.character import RecursiveCharacterTextSplitter

# load_dotenv()

# def load_chunks():
#     texts = []
#     for path in glob.glob("data/*.vtt"):
#         with open(path) as f:
#             lines = [ln.strip() for ln in f
#                      if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
#         texts.append(" ".join(lines))
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     return splitter.split_text("\n\n".join(texts))


# chunks = load_chunks()
# sample = random.sample(chunks, min(15, len(chunks)))
# contexts = [[c] for c in sample]

# synthesizer = Synthesizer(model="llama-3.3-70b-versatile")
# goldens = synthesizer.generate_goldens_from_contexts(
#     contexts=contexts,
#     include_expected_output=True,
#     max_goldens_per_context=1,
# )

# rows = []
# for i, g in enumerate(goldens, 1):
#     rows.append({
#         "id": f"g{i:03d}",
#         "query": g.input,
#         "ideal_answer": g.expected_output,
#         "source": "TODO-verify",
#     })

# with open("goldens/retriever_deepeval_goldens.json", "w") as f:
#     json.dump(rows, f, indent=2, ensure_ascii=False)

# print(f"wrote {len(rows)} DRAFT goldens -> goldens/component_goldens_draft.json")
# print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")


import os
import glob
import json
import random

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from groq import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()


class GroqModel(DeepEvalBaseLLM):
    """Wraps a Groq-hosted model so DeepEval's Synthesizer can call it."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self.model

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=5, max=90),
        stop=stop_after_attempt(8),
    )
    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=5, max=90),
        stop=stop_after_attempt(8),
    )
    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return f"Groq {self.model_name}"


def load_chunks():
    texts = []
    for path in glob.glob("data/*.vtt"):
        with open(path) as f:
            lines = [
                ln.strip() for ln in f
                if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln
            ]
        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))


def main():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found in data/*.vtt — check your data folder.")

    sample = random.sample(chunks, min(15, len(chunks)))
    contexts = [[c] for c in sample]

    synthesizer = Synthesizer(
        model=GroqModel(model_name="llama-3.3-70b-versatile"),
        max_concurrent=1,  # fully sequential — safest for Groq free-tier TPM limits
    )
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=1,
    )

    rows = []
    for i, g in enumerate(goldens, 1):
        rows.append({
            "id": f"g{i:03d}",
            "query": g.input,
            "ideal_answer": g.expected_output,
            "source": "TODO-verify",
        })

    os.makedirs("goldens", exist_ok=True)
    out_path = "goldens/retriever_deepeval_goldens.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(rows)} DRAFT goldens -> {out_path}")
    print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")


if __name__ == "__main__":
    main()