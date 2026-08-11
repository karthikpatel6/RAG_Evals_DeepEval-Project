import json
import os
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "300"

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from src.retriever import build_retriever
from goldens.goldens_generator import GroqModel

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL_NAME = "llama-3.3-70b-versatile"
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

retriever = build_retriever()
test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )

judge_model = GroqModel(model_name=JUDGE_MODEL_NAME)

metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=judge_model, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=judge_model, include_reason=True),
]

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "base_k5",
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL_NAME,
        "golden_set": GOLDEN_PATH,
    },
    async_config=AsyncConfig(
        run_async=True,
        max_concurrent=1,
        throttle_value=2,
    ),
)