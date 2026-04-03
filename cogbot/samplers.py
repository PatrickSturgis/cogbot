"""LLM sampler backends for CogBot pipelines."""

import time

import pandas as pd


class OpenAISampler:
    """OpenAI API sampler. Works with any model available via the OpenAI API.

    Args:
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY env var.
        model: Model identifier (default: "gpt-4o").
        temperature: Sampling temperature (default: 0.7).
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o",
                 temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def query_single(self, system_prompt: str, user_prompt: str,
                     max_tokens: int = 1000, temperature: float = None) -> str:
        """Query the model with a single system/user prompt pair."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        temp = temperature if temperature is not None else self.temperature

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temp,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
