"""CogBot evaluation pipelines for survey question pretesting."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .prompts import (
    COGTEST_ANALYST_SYSTEM,
    COGTEST_ANALYST_USER,
    COGTEST_RESPONDENT_SYSTEM,
    COGTEST_RESPONDENT_USER,
    COGTEST_SYNTHESIS_SYSTEM,
    COGTEST_SYNTHESIS_USER,
    EXPERT_REVIEW_SYSTEM,
    EXPERT_REVIEW_USER,
    EXPERT_SYNTHESIS_SYSTEM,
    EXPERT_SYNTHESIS_USER,
)


def _call_with_retry(fn, max_retries=3):
    """Call *fn* with retry and exponential backoff for rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            if attempt < max_retries - 1 and (
                "rate" in err_str or "limit" in err_str or "429" in err_str
            ):
                time.sleep(2 ** (attempt + 1))
                continue
            raise


class CogTestPipeline:
    """Structured cognitive interview pipeline (Tourangeau response model).

    Three-stage process:

    1. **Respondent think-aloud** -- simulated respondents work through the
       question following the Tourangeau stages (comprehension, retrieval,
       judgement, response mapping, confidence).
    2. **Analyst coding** -- each transcript is independently analysed for
       problems, with severity ratings on a 1--10 scale.
    3. **Cross-respondent synthesis** -- patterns are aggregated into a
       summary of distinct problems.

    Works with any sampler that exposes a
    ``query_single(system, user, max_tokens)`` method.

    Args:
        sampler: LLM sampler instance (e.g. :class:`cogbot.OpenAISampler`).
        max_workers: Maximum concurrent API calls (default 3).
    """

    def __init__(self, sampler, max_workers: int = 3):
        self.sampler = sampler
        self.max_workers = max_workers

    def run(self, df: pd.DataFrame, question_text: str,
            response_options: str = "", stage_callback=None) -> dict:
        """Run the full cognitive interview pipeline.

        Args:
            df: DataFrame with a ``backstory`` column (one row per
                simulated respondent).
            question_text: The survey question being tested.
            response_options: Response option text.
            stage_callback: Optional ``fn(stage_name, detail)`` for progress.

        Returns:
            dict with keys ``transcripts``, ``analyses``, ``synthesis``,
            ``confidence_ratings``, ``errors``, ``metadata``.
        """
        # Collect valid backstories
        respondents = []
        for idx, row in df.iterrows():
            bs = row.get("backstory", "")
            if pd.isna(bs) or str(bs).strip() == "":
                continue
            respondents.append({"idx": idx, "backstory": str(bs)})

        n = len(respondents)
        if n == 0:
            return {
                "transcripts": [], "analyses": [], "synthesis": None,
                "confidence_ratings": [], "errors": [], "metadata": {"n": 0},
            }

        # --- Stage 1: Respondent think-alouds ---
        if stage_callback:
            stage_callback("Stage 1", f"Generating {n} think-aloud transcripts...")

        transcripts = [None] * n
        errors = []

        def _call_respondent(i):
            user_prompt = COGTEST_RESPONDENT_USER.format(
                backstory=respondents[i]["backstory"],
                question_text=question_text,
                response_options=response_options,
            )
            text = _call_with_retry(
                lambda: self.sampler.query_single(
                    COGTEST_RESPONDENT_SYSTEM, user_prompt, max_tokens=1024
                )
            )
            return i, text

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_call_respondent, i) for i in range(n)]
            for future in as_completed(futures):
                try:
                    i, text = future.result()
                    transcripts[i] = text
                except Exception as e:
                    errors.append(f"Stage 1: {str(e)[:100]}")

        transcript_records = []
        for i, resp in enumerate(respondents):
            transcript_records.append({
                "respondent_id": f"r{i+1}",
                "idx": resp["idx"],
                "backstory": resp["backstory"],
                "transcript": transcripts[i] or "[ERROR: No response]",
            })

        # --- Stage 2a: Individual transcript analysis ---
        if stage_callback:
            stage_callback("Stage 2a", f"Analysing {n} transcripts...")

        analyses = [None] * n

        def _call_analyst(i):
            rec = transcript_records[i]
            if rec["transcript"].startswith("[ERROR"):
                return i, {"problems": [], "respondent_id": rec["respondent_id"]}
            user_prompt = COGTEST_ANALYST_USER.format(
                question_text=question_text,
                response_options=response_options,
                transcript=rec["transcript"],
                respondent_id=rec["respondent_id"],
            )
            raw = _call_with_retry(
                lambda: self.sampler.query_single(
                    COGTEST_ANALYST_SYSTEM, user_prompt, max_tokens=1024
                )
            )
            try:
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                return i, json.loads(cleaned.strip())
            except (json.JSONDecodeError, ValueError):
                return i, {"raw": raw, "parse_error": True}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_call_analyst, i) for i in range(n)]
            for future in as_completed(futures):
                try:
                    i, parsed = future.result()
                    analyses[i] = parsed
                except Exception as e:
                    errors.append(f"Stage 2a: {str(e)[:100]}")

        analyses = [
            a if a is not None else {"parse_error": True, "raw": "Error"}
            for a in analyses
        ]

        # --- Stage 2b: Cross-respondent synthesis ---
        if stage_callback:
            stage_callback("Stage 2b", "Synthesising patterns across respondents...")

        analyses_parts = []
        for i, rec in enumerate(transcript_records):
            a = analyses[i]
            if isinstance(a, dict) and "parse_error" not in a:
                problems = a.get("problems", [])
                if problems:
                    prob_lines = []
                    for p in problems:
                        sev = p.get("severity", "?")
                        prob_lines.append(
                            f"  - {p.get('type', 'UNKNOWN')} (severity: {sev}/10): "
                            f"{p.get('description', '')} "
                            f"[Evidence: {p.get('evidence', '')}]"
                        )
                    analyses_parts.append(
                        f"RESPONDENT {rec['respondent_id']}:\n"
                        + "\n".join(prob_lines)
                    )
                else:
                    analyses_parts.append(
                        f"RESPONDENT {rec['respondent_id']}:\n"
                        "  No problems identified."
                    )
            else:
                analyses_parts.append(
                    f"RESPONDENT {rec['respondent_id']}:\n"
                    "  [Analysis could not be parsed]"
                )

        analyses_block = "\n\n---\n\n".join(analyses_parts)

        synthesis_user = COGTEST_SYNTHESIS_USER.format(
            question_text=question_text,
            response_options=response_options,
            analyses_block=analyses_block,
        )

        try:
            raw_synthesis = _call_with_retry(
                lambda: self.sampler.query_single(
                    COGTEST_SYNTHESIS_SYSTEM, synthesis_user, max_tokens=2048
                )
            )
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_synthesis)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            synthesis = json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            synthesis = {"raw": raw_synthesis, "parse_error": True}
        except Exception as e:
            synthesis = {"error": str(e), "parse_error": True}
            errors.append(f"Stage 2b: {str(e)[:100]}")

        # --- Extract confidence ratings from transcripts ---
        confidence_ratings = []
        for rec in transcript_records:
            transcript = rec["transcript"]
            conf_section = re.search(
                r"(?i)\bCONFIDENCE\b[:\s]*(.*?)(?:\n\s*\d+\.\s|\Z)",
                transcript, re.DOTALL,
            )
            if conf_section:
                section = conf_section.group(1)
                rating_match = (
                    re.search(r"([1-5])\s*/\s*5", section)
                    or re.search(r"([1-5])\s+out of\s+5", section)
                    or re.search(r"(?:^|[^\d-])([1-5])(?:[^\d-]|$)", section)
                )
                if rating_match:
                    rating = int(rating_match.group(1))
                    confidence_ratings.append({
                        "respondent_id": rec["respondent_id"],
                        "rating": rating,
                    })

        return {
            "transcripts": transcript_records,
            "analyses": analyses,
            "synthesis": synthesis,
            "confidence_ratings": confidence_ratings,
            "errors": errors,
            "metadata": {
                "n": n,
                "question_text": question_text,
                "response_options": response_options,
            },
        }


class ExpertReviewPipeline:
    """Expert review pipeline: 3 independent LLM experts evaluate a question
    against a directive checklist of known problem types, then a synthesis
    step aggregates their reviews.

    Only 4 API calls total (3 experts + 1 synthesis). No respondent
    simulation, backstories, or sample size needed.

    Works with any sampler that exposes a
    ``query_single(system, user, max_tokens, temperature)`` method.

    Args:
        sampler: LLM sampler instance (e.g. :class:`cogbot.OpenAISampler`).
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def run(self, question_text: str, response_options: str = "",
            stage_callback=None) -> dict:
        """Run the expert review pipeline.

        Args:
            question_text: The survey question being evaluated.
            response_options: Response option text.
            stage_callback: Optional ``fn(stage_name, detail)`` for progress.

        Returns:
            dict with keys ``expert_reviews``, ``synthesis``, ``errors``,
            ``metadata``.
        """
        errors = []

        # --- Stage 1: 3 independent expert reviews ---
        if stage_callback:
            stage_callback("Stage 1", "Expert reviews: 0/3 complete...")

        expert_reviews = []
        for i in range(3):
            expert_id = f"e{i+1}"
            user_prompt = EXPERT_REVIEW_USER.format(
                question_text=question_text,
                response_options=response_options,
                expert_id=expert_id,
            )
            try:
                raw = _call_with_retry(
                    lambda: self.sampler.query_single(
                        EXPERT_REVIEW_SYSTEM, user_prompt,
                        max_tokens=1024, temperature=0.7,
                    )
                )
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                parsed = json.loads(cleaned.strip())
                expert_reviews.append(parsed)
            except (json.JSONDecodeError, ValueError):
                expert_reviews.append({
                    "expert_id": expert_id, "raw": raw, "parse_error": True,
                })
                errors.append(f"Stage 1 expert {expert_id}: JSON parse error")
            except Exception as e:
                expert_reviews.append({
                    "expert_id": expert_id, "error": str(e), "parse_error": True,
                })
                errors.append(f"Stage 1 expert {expert_id}: {str(e)[:100]}")

            if stage_callback:
                stage_callback("Stage 1", f"Expert reviews: {i+1}/3 complete...")

        # --- Stage 2: Synthesis ---
        if stage_callback:
            stage_callback("Stage 2", "Synthesising expert reviews...")

        reviews_parts = []
        for i, review in enumerate(expert_reviews):
            expert_id = f"e{i+1}"
            if isinstance(review, dict) and "parse_error" not in review:
                problems = review.get("problems", [])
                no_problems = review.get("no_problems", [])
                if problems:
                    prob_lines = []
                    for p in problems:
                        sev = p.get("severity", "?")
                        prob_lines.append(
                            f"  - {p.get('type', 'UNKNOWN')} (severity: {sev}/10): "
                            f"{p.get('description', '')} "
                            f"[Evidence: {p.get('evidence', '')}]"
                        )
                    reviews_parts.append(
                        f"EXPERT {expert_id}:\nProblems found:\n"
                        + "\n".join(prob_lines)
                        + (f"\nAbsent: {', '.join(no_problems)}"
                           if no_problems else "")
                    )
                else:
                    reviews_parts.append(
                        f"EXPERT {expert_id}:\n  No problems identified."
                    )
            else:
                reviews_parts.append(
                    f"EXPERT {expert_id}:\n  [Review could not be parsed]"
                )

        reviews_block = "\n\n---\n\n".join(reviews_parts)

        synthesis_user = EXPERT_SYNTHESIS_USER.format(
            question_text=question_text,
            response_options=response_options,
            reviews_block=reviews_block,
        )

        try:
            raw_synthesis = _call_with_retry(
                lambda: self.sampler.query_single(
                    EXPERT_SYNTHESIS_SYSTEM, synthesis_user,
                    max_tokens=2048, temperature=0.3,
                )
            )
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_synthesis)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            synthesis = json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            synthesis = {"raw": raw_synthesis, "parse_error": True}
        except Exception as e:
            synthesis = {"error": str(e), "parse_error": True}
            errors.append(f"Stage 2: {str(e)[:100]}")

        return {
            "expert_reviews": expert_reviews,
            "synthesis": synthesis,
            "errors": errors,
            "metadata": {
                "n_experts": 3,
                "question_text": question_text,
                "response_options": response_options,
            },
        }
