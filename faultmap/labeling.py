from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .llm import AsyncLLMClient


@dataclass(frozen=True)
class ClusterLabel:
    name: str                    # 2-5 word name, e.g. "Date Formatting Queries"
    description: str             # One sentence description
    root_cause: str = ""         # Why the LLM fails on this slice (failure slices only)
    suggested_remediation: str = ""  # Concrete system-prompt fix (failure slices only)


MAX_EXAMPLES_FOR_NAMING = 15
MAX_CHARS_PER_EXAMPLE = 300


async def label_cluster(
    client: AsyncLLMClient,
    representative_texts: list[str],
    context: str = "failure slice",
) -> ClusterLabel:
    """
    Ask the LLM to name a single cluster.

    Algorithm:
    1. Truncate texts to MAX_EXAMPLES_FOR_NAMING, each to MAX_CHARS_PER_EXAMPLE
    2. System prompt: explain task, request "Name: ...\nDescription: ..." format
    3. Call client.complete() with temperature=0
    4. Parse response; fallback: first line = name, rest = description

    Edge cases:
    - LLM doesn't follow format → graceful fallback parsing
    - Very short texts → still works, just less context for naming
    """
    truncated = [
        t[:MAX_CHARS_PER_EXAMPLE] + ("..." if len(t) > MAX_CHARS_PER_EXAMPLE else "")
        for t in representative_texts[:MAX_EXAMPLES_FOR_NAMING]
    ]

    examples_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(truncated))

    if context == "failure slice":
        system_prompt = (
            "You are an expert AI debugger analyzing a cluster of similar text inputs "
            "where an LLM is systematically failing. Given the example inputs below, provide:\n"
            "1. A concise name (2-5 words) that captures the common theme\n"
            "2. A one-sentence description of what these texts have in common\n"
            "3. A root cause analysis: why is the LLM likely failing on this type of input?\n"
            "4. A suggested fix: a concrete 1-2 sentence addition or modification to the "
            "system prompt that would address this failure pattern\n\n"
            "Respond in exactly this format:\n"
            "Name: <your name>\n"
            "Description: <your description>\n"
            "Root Cause: <your root cause analysis>\n"
            "Suggested Fix: <your suggested remediation>"
        )
        max_tokens = 400
    else:
        system_prompt = (
            f"You are analyzing a cluster of similar text inputs that form a {context}. "
            f"Given the example texts below, provide:\n"
            f"1. A concise name (2-5 words) that captures the common theme\n"
            f"2. A one-sentence description of what these texts have in common\n\n"
            f"Respond in exactly this format:\n"
            f"Name: <your name>\n"
            f"Description: <your description>"
        )
        max_tokens = 150

    user_prompt = f"Here are the example texts from this cluster:\n\n{examples_text}"

    response = await client.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    return _parse_label_response(response)


def _parse_label_response(response: str) -> ClusterLabel:
    """
    Parse "Name: ...\nDescription: ..." format, with optional Root Cause / Suggested Fix.
    Fallback: first line = name, rest = description.
    """
    lines = response.strip().split("\n")
    name = ""
    description = ""
    root_cause = ""
    suggested_remediation = ""

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("name:"):
            name = stripped[len("name:"):].strip()
        elif stripped.lower().startswith("description:"):
            description = stripped[len("description:"):].strip()
        elif stripped.lower().startswith("root cause:"):
            root_cause = stripped[len("root cause:"):].strip()
        elif stripped.lower().startswith("suggested fix:"):
            suggested_remediation = stripped[len("suggested fix:"):].strip()

    if not name:
        name = lines[0].strip()[:80]
    if not description:
        description = " ".join(lines[1:]).strip()[:200] if len(lines) > 1 else name

    return ClusterLabel(
        name=name,
        description=description,
        root_cause=root_cause,
        suggested_remediation=suggested_remediation,
    )


async def label_clusters(
    client: AsyncLLMClient,
    clusters_texts: list[list[str]],
    context: str = "failure slice",
) -> list[ClusterLabel]:
    """
    Label multiple clusters concurrently via asyncio.gather.
    Returns labels in same order as clusters_texts.
    """
    tasks = [
        label_cluster(client, texts, context=context)
        for texts in clusters_texts
    ]
    return list(await asyncio.gather(*tasks))
