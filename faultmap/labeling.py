from __future__ import annotations
import asyncio
from dataclasses import dataclass

from .llm import AsyncLLMClient


@dataclass(frozen=True)
class ClusterLabel:
    name: str          # 2-5 word name, e.g. "Date Formatting Queries"
    description: str   # One sentence description


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

    system_prompt = (
        f"You are analyzing a cluster of similar text inputs that form a {context}. "
        f"Given the example texts below, provide:\n"
        f"1. A concise name (2-5 words) that captures the common theme\n"
        f"2. A one-sentence description of what these texts have in common\n\n"
        f"Respond in exactly this format:\n"
        f"Name: <your name>\n"
        f"Description: <your description>"
    )

    user_prompt = f"Here are the example texts from this cluster:\n\n{examples_text}"

    response = await client.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    return _parse_label_response(response)


def _parse_label_response(response: str) -> ClusterLabel:
    """
    Parse "Name: ...\nDescription: ..." format.
    Fallback: first line = name, rest = description.
    """
    lines = response.strip().split("\n")
    name = ""
    description = ""

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("name:"):
            name = stripped[len("name:"):].strip()
        elif stripped.lower().startswith("description:"):
            description = stripped[len("description:"):].strip()

    if not name:
        name = lines[0].strip()[:80]
    if not description:
        description = " ".join(lines[1:]).strip()[:200] if len(lines) > 1 else name

    return ClusterLabel(name=name, description=description)


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
