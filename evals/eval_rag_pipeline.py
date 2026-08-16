import json
from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = "llama-3.3-70b-versatile"
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["query"])

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],
            retrieval_context=result["context"],
        )
    )

metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]

# 4. EVALUATE
evaluate(test_cases=test_cases, metrics=metrics)