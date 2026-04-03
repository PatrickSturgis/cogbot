# CogBot

LLM-based cognitive interviewing for survey question pretesting.

CogBot uses large language models to evaluate draft survey questions for design problems such as double-barrelled wording, presupposition failures, comprehension difficulties, and response mapping issues. It implements two complementary evaluation pipelines that can be run with any OpenAI-compatible model.

## Installation

```bash
git clone https://github.com/PatrickSturgis/cogbot.git
cd cogbot
pip install -e .
```

## Quick start

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### Expert review pipeline

Three independent LLM experts evaluate the question against a checklist of known problem types, then a synthesis step aggregates their findings. Fast (4 API calls).

```python
from cogbot import ExpertReviewPipeline, OpenAISampler

sampler = OpenAISampler(model="gpt-4o")
pipeline = ExpertReviewPipeline(sampler)

result = pipeline.run(
    question_text="How often do you use the internet for personal or work purposes?",
    response_options="Every day / Several times a week / Once a week / Less often / Never",
)

for problem in result["synthesis"]["problems_detected"]:
    print(f"[{problem['type']}] severity {problem['mean_severity']}: {problem['description']}")
```

### Cognitive interview pipeline

Simulated respondents work through the question via think-aloud protocols, then an analyst codes each transcript for problems. Richer output but slower (~60 API calls for n=30).

```python
import pandas as pd
from cogbot import CogTestPipeline, OpenAISampler, load_backstories

sampler = OpenAISampler(model="gpt-4o")
pipeline = CogTestPipeline(sampler, max_workers=3)

backstories = load_backstories(n=30, backstory_type="short", seed=42)
df = pd.DataFrame({"backstory": backstories})

result = pipeline.run(
    df=df,
    question_text="How often do you use the internet for personal or work purposes?",
    response_options="Every day / Several times a week / Once a week / Less often / Never",
)

for problem in result["synthesis"]["problems_detected"]:
    print(f"[{problem['type']}] severity {problem['mean_severity']}, "
          f"{len(problem['respondents_affected'])} respondents: {problem['description']}")
```

## How it works

### Expert review pipeline

1. Three LLM "experts" independently evaluate the question against seven problem types (comprehension, double-barrelled, retrieval, response mapping, social desirability, presupposition, tautology).
2. Each expert rates severity on a 1--10 scale for problems found.
3. A synthesis step aggregates across experts, reporting which experts flagged each problem and the mean severity.

### Cognitive interview pipeline

1. Simulated respondents (given demographic backstories from ESS microdata) think aloud through the Tourangeau response model stages: comprehension, retrieval, judgement, response mapping, and confidence.
2. An analyst LLM codes each transcript for problems with evidence citations and severity ratings.
3. A synthesis step identifies patterns across respondents, reporting prevalence and mean severity.

### Backstories

The `data/` directory contains ~2,200 respondent profiles derived from European Social Survey (ESS) Round 11 UK microdata. Each profile includes demographic attributes (age, gender, education, employment, household composition). The "long" variant adds attitudinal variables. These provide diverse, realistic respondent personas for the cognitive interview pipeline.

## Output format

Both pipelines return a dict. The `synthesis` key contains the aggregated results:

```json
{
  "problems_detected": [
    {
      "type": "DOUBLE-BARRELLED",
      "description": "The question asks about internet use for two distinct purposes...",
      "mean_severity": 7.0,
      "respondents_affected": ["r1", "r5", "r12"],
      "evidence_summary": "..."
    }
  ]
}
```

## Using other models

Pass any model name supported by the OpenAI API:

```python
sampler = OpenAISampler(model="gpt-4o-mini")
```

For non-OpenAI providers with OpenAI-compatible APIs, you can subclass `OpenAISampler` or write a custom sampler with a `query_single(system_prompt, user_prompt, max_tokens, temperature)` method.

## License

MIT
