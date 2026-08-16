"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault — the context was already correct.

    python -m evals.eval_generator
"""

import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate
load_dotenv()
from goldens.goldens_generator import GroqModel

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL =  GroqModel(model_name="llama-3.3-70b-versatile")
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

test_cases = []
for g in goldens:
    context = g["ideal_context"]
    answer = generate(g["query"], context)

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=context,
        )
    )

metrics = [FaithfulnessMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=True,
),
AnswerRelevancyMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=True
)]

evaluate(test_cases=test_cases, metrics=metrics)