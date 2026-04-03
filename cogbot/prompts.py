"""Prompt templates for the CogBot cognitive interview and expert review pipelines."""

# ============================================================================
# COGNITIVE INTERVIEW PIPELINE PROMPTS
# Structured think-aloud (Tourangeau model) + analyst coding + synthesis
# ============================================================================

COGTEST_RESPONDENT_SYSTEM = """You are this person. Respond as yourself using first person.
Do not invent details beyond the provided attributes.
You are taking part in a cognitive interview about a survey question.
Work through the question naturally, thinking aloud as you go."""

COGTEST_RESPONDENT_USER = """WHO YOU ARE:
{backstory}

SURVEY QUESTION:
{question_text}

RESPONSE OPTIONS: {response_options}

Think aloud as you work through answering this question:

1. COMPREHENSION: Put the question in your own words. What is it asking you?
2. RETRIEVAL: What from your life or experience are you drawing on?
3. JUDGEMENT: Are you having to weigh up or combine different things?
4. RESPONSE MAPPING: Which option fits best? How well does it capture what you want to say? Is anything missing?
5. CHOSEN ANSWER: State your final answer.
6. CONFIDENCE: Rate 1-5 how confident you are in that answer."""

COGTEST_ANALYST_SYSTEM = """You are an expert in cognitive interviewing and survey methodology.
You are reviewing a think-aloud transcript from a cognitive interview.
Your job is to identify problems with the question based on evidence in the transcript."""

COGTEST_ANALYST_USER = """SURVEY QUESTION:
{question_text}

RESPONSE OPTIONS: {response_options}

RESPONDENT THINK-ALOUD TRANSCRIPT:
{transcript}

Identify ONLY problems that are clearly evidenced in this transcript. Do NOT go looking for problems that are not there. Most transcripts will have only 1-2 genuine problems, and some will have none. If the respondent navigated the question without difficulty, return an empty problems list.

A problem must be supported by specific evidence in the transcript - something the respondent actually said or demonstrably struggled with. Do not infer problems that the respondent did not experience.

When a genuine problem is found, classify it using whichever of these categories fits best:
- COMPREHENSION PROBLEMS: Respondent misunderstood or misinterpreted terms
- DOUBLE-BARRELLED: Question asks about two distinct things requiring different answers
- RETRIEVAL DIFFICULTIES: Respondent struggled to recall or estimate
- RESPONSE MAPPING FAILURES: Answer did not fit the available options
- SOCIAL DESIRABILITY / SENSITIVITY: Respondent hedged or showed discomfort
- PRESUPPOSITION FAILURES: Question assumed something untrue for this respondent
- TAUTOLOGY / LOGICAL PROBLEMS: Circularity between question wording and response options

If the problem does not fit any category above, use a short descriptive label.

For each problem:
- Cite specific evidence from the transcript (what the respondent actually said)
- Rate SEVERITY 1-10: the degree to which this problem would distort the accuracy of the answer (1 = negligible distortion, 10 = answer rendered meaningless)

Return ONLY valid JSON, no markdown fencing:
{{"respondent_id": "{respondent_id}", "problems": [{{"type": "...", "description": "...", "evidence": "...", "severity": 7}}]}}"""

COGTEST_SYNTHESIS_SYSTEM = """You are a senior survey methodologist writing a cognitive testing report.
You are synthesising the results of individual transcript analyses from a cognitive interview study."""

COGTEST_SYNTHESIS_USER = """SURVEY QUESTION:
{question_text}

RESPONSE OPTIONS: {response_options}

Below are the coded problems identified by an analyst for each respondent's think-aloud transcript.
Each problem has a severity score (1-10) indicating the degree to which it would distort accuracy of the answer.
Your job is to aggregate these into a summary of the distinct problems found across respondents.
Use the SAME problem type labels as the individual analyses (e.g. if analysts coded "DOUBLE-BARRELLED", use that label, not a paraphrase).

INDIVIDUAL ANALYSES:
{analyses_block}

For each distinct problem type that was identified:
- Use the same type label as in the individual analyses
- List which respondents showed evidence of it
- Calculate the mean severity score across respondents who had this problem
- Summarise the evidence across respondents

Return ONLY valid JSON, no markdown fencing:
{{"question_id": "synthesis", "problems_detected": [{{"type": "...", "description": "...", "respondents_affected": [...], "mean_severity": 6.5, "evidence_summary": "..."}}]}}"""


# ============================================================================
# EXPERT REVIEW PIPELINE PROMPTS
# Directive checklist approach: 3 independent expert reviewers + synthesis
# ============================================================================

EXPERT_REVIEW_SYSTEM = """You are an expert in survey methodology and questionnaire design.
You are reviewing a draft survey question for potential cognitive problems
that respondents might experience when trying to answer it.
Evaluate the question systematically against known problem types."""

EXPERT_REVIEW_USER = """SURVEY QUESTION:
{question_text}

RESPONSE OPTIONS: {response_options}

Evaluate this question for evidence of each of the following problem types:

1. COMPREHENSION PROBLEMS: Are any words or phrases ambiguous, technical,
   or likely to be misunderstood? Could respondents interpret the question
   differently from its intended meaning?
2. DOUBLE-BARRELLED: Does the question ask about two or more distinct things
   at once for which respondents might want to give different answers?
3. RETRIEVAL DIFFICULTIES: Does the question require respondents to recall
   information that may be difficult to retrieve accurately? Is the reference
   period too long, too vague, or inappropriate?
4. RESPONSE MAPPING FAILURES: Do the response options adequately capture the
   range of possible answers? Are options missing, overlapping, or ambiguous?
   Can respondents map their intended answer to the available options?
5. SOCIAL DESIRABILITY / SENSITIVITY: Is the question likely to produce
   socially desirable responding or discomfort?
6. PRESUPPOSITION FAILURES: Does the question assume something that may not
   be true for all respondents?
7. TAUTOLOGY / LOGICAL PROBLEMS: Is there circularity between the question
   wording and the response options?

This list is not exhaustive. If you identify any other problems with the
question that do not fit the categories above, include them using a short
descriptive label.

For each problem type listed above, state whether it is PRESENT or ABSENT.
If PRESENT: describe the specific problem, cite the relevant words or
features of the question, and rate SEVERITY 1-10 (1 = negligible impact
on data quality, 10 = answer rendered meaningless).
If ABSENT: briefly explain why.

Return ONLY valid JSON, no markdown fencing:
{{"expert_id": "{expert_id}", "problems": [{{"type": "...",
"description": "...", "evidence": "...", "severity": 7}}],
"no_problems": ["SOCIAL DESIRABILITY / SENSITIVITY", ...]}}"""

EXPERT_SYNTHESIS_SYSTEM = """You are a senior survey methodologist writing a question evaluation report.
You are synthesising the results of independent expert reviews of a survey question."""

EXPERT_SYNTHESIS_USER = """SURVEY QUESTION:
{question_text}

RESPONSE OPTIONS: {response_options}

Below are the independent evaluations from 3 expert reviewers.
Each identified problem has a severity score (1-10).
Your job is to aggregate these into a summary of the distinct problems found.
Use the SAME problem type labels as the individual reviews.

EXPERT REVIEWS:
{reviews_block}

For each distinct problem that was identified:
- Use the same type label as in the individual reviews
- List which experts flagged it
- Calculate the mean severity score across experts who flagged it
- Summarise the evidence

Return ONLY valid JSON, no markdown fencing:
{{"question_id": "synthesis", "problems_detected": [{{"type": "...",
"description": "...", "experts_affected": ["e1", "e2"],
"mean_severity": 6.5, "evidence_summary": "..."}}]}}"""
