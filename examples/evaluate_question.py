"""Minimal example: evaluate a survey question with both CogBot pipelines."""

import json
import os

import pandas as pd

from cogbot import (
    CogTestPipeline,
    ExpertReviewPipeline,
    OpenAISampler,
    load_backstories,
)

# -- Configuration --
QUESTION = (
    "Some people think that immigration is good for the economy, while "
    "others think it places too much pressure on public services. On the "
    "whole, do you think immigration has been good or bad for the economy?"
)
RESPONSE_OPTIONS = (
    "Very good / Quite good / Neither good nor bad / Quite bad / Very bad"
)

# -- Set up the sampler --
sampler = OpenAISampler(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.7,
)


def progress(stage, detail):
    print(f"  [{stage}] {detail}")


# -- 1. Expert Review Pipeline (fast: 4 API calls) --
print("=" * 60)
print("EXPERT REVIEW PIPELINE")
print("=" * 60)

er_pipeline = ExpertReviewPipeline(sampler)
er_result = er_pipeline.run(QUESTION, RESPONSE_OPTIONS, stage_callback=progress)

synthesis = er_result["synthesis"]
if "problems_detected" in synthesis:
    for p in synthesis["problems_detected"]:
        print(f"  [{p.get('type')}] severity {p.get('mean_severity')}: "
              f"{p.get('description', '')[:80]}")
else:
    print("  No problems detected (or parse error).")

print(f"\n  Errors: {er_result['errors'] or 'none'}")

# -- 2. Cognitive Interview Pipeline (slower: ~60+ API calls for n=30) --
print("\n" + "=" * 60)
print("COGNITIVE INTERVIEW PIPELINE (n=10 respondents)")
print("=" * 60)

backstories = load_backstories(n=10, backstory_type="short", seed=42)
df = pd.DataFrame({"backstory": backstories})

ci_pipeline = CogTestPipeline(sampler, max_workers=3)
ci_result = ci_pipeline.run(df, QUESTION, RESPONSE_OPTIONS, stage_callback=progress)

synthesis = ci_result["synthesis"]
if "problems_detected" in synthesis:
    for p in synthesis["problems_detected"]:
        affected = p.get("respondents_affected", [])
        print(f"  [{p.get('type')}] severity {p.get('mean_severity')}, "
              f"{len(affected)} respondents: "
              f"{p.get('description', '')[:80]}")
else:
    print("  No problems detected (or parse error).")

confidence = ci_result["confidence_ratings"]
if confidence:
    ratings = [c["rating"] for c in confidence]
    print(f"\n  Mean confidence: {sum(ratings)/len(ratings):.1f}/5 "
          f"(n={len(ratings)})")

print(f"\n  Errors: {ci_result['errors'] or 'none'}")
