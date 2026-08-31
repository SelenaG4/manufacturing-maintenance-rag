"""Retrieval-augmented answering: retrieve the most relevant maintenance-guide
chunks, then produce a grounded answer that cites its sources.

Generation has a three-tier fallback, the same graceful pattern as this
portfolio's swiss-claims-assistant:
  1. Azure OpenAI  (if AZURE_OPENAI_* env vars are set),
  2. OpenAI        (if OPENAI_API_KEY is set),
  3. offline extractive  (default) -- synthesizes the answer directly from the
     retrieved passages, no API key required, so the service is fully functional
     and safe to demo with zero configuration.

A grounding guardrail comes first in every mode: if nothing in the corpus is
relevant enough, the assistant says so instead of inventing an answer -- the
whole point of RAG is to answer *from the documents*, not from thin air.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from app import telemetry
from app.config import TOP_K
from app.corpus import Chunk
from app.index import RagIndex

MIN_SCORE = 0.10  # below this, treat the query as out-of-scope for the corpus


@dataclass
class Source:
    chunk_id: str
    doc_title: str
    heading: str
    source_file: str
    score: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[Source]
    mode: str          # "azure_openai" | "openai" | "offline_extractive" | "no_answer"
    grounded: bool


def _sources(hits: list[tuple[Chunk, float]]) -> list[Source]:
    return [Source(c.chunk_id, c.doc_title, c.heading, c.source_file, round(s, 4)) for c, s in hits]


def _context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i+1}] {c.doc_title} — {c.heading}\n{c.text.split(chr(10), 1)[-1]}"
                       for i, (c, _) in enumerate(hits))


def _offline_answer(query: str, hits: list[tuple[Chunk, float]]) -> str:
    """Extractive grounding: lead with the single best passage's guidance, then
    point to the other retrieved sections. No LLM, but still a useful, cited
    answer drawn entirely from the maintenance guides."""
    top_chunk, _ = hits[0]
    body = top_chunk.text.split("\n", 1)[-1].strip()
    lead = f"From **{top_chunk.doc_title} — {top_chunk.heading}**:\n\n{body}"
    others = [f"- {c.doc_title} — {c.heading}" for c, _ in hits[1:3]]
    if others:
        lead += "\n\nRelated sections you may also want:\n" + "\n".join(others)
    return lead


def _llm_answer(query: str, hits: list[tuple[Chunk, float]]) -> tuple[str, str] | None:
    """Try Azure OpenAI then OpenAI; return (answer, mode) or None to fall back."""
    context = _context(hits)
    system = ("You are a manufacturing-maintenance assistant. Answer the question using ONLY the "
              "context passages provided. Be concise and practical. If the context does not contain "
              "the answer, say so. Cite the passages you used by their [n] number.")
    user = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model, temperature=0.1,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return resp.choices[0].message.content.strip(), "azure_openai"
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.1,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return resp.choices[0].message.content.strip(), "openai"
    except Exception:
        return None  # any error -> graceful fallback to offline
    return None


def answer_question(index: RagIndex, query: str, k: int = TOP_K) -> RagAnswer:
    # Retrieval and generation are traced separately on purpose: they fail and
    # slow down for entirely different reasons, and an end-to-end request
    # duration cannot tell you which one moved. See app/telemetry.py.
    with telemetry.span("rag.retrieve", **{"rag.k": k}) as retrieve_span:
        hits = index.search(query, k)
        top_score = hits[0][1] if hits else 0.0
        telemetry.set_attributes(
            retrieve_span,
            **{"rag.hit_count": len(hits), "rag.top_score": float(top_score)},
        )

    if not hits or top_score < MIN_SCORE:
        # The guardrail firing is a product event, not an error -- it is the
        # service correctly refusing to answer off-corpus. Recorded so the rate
        # can be watched: a sudden climb means users are asking about machines
        # the knowledge base does not cover yet.
        with telemetry.span(
            "rag.guardrail_refused",
            **{"rag.top_score": float(top_score), "rag.min_score": MIN_SCORE},
        ):
            pass
        return RagAnswer(
            answer=("I couldn't find anything relevant in the maintenance guides for that. "
                    "Try rephrasing, or ask about a specific machine, symptom, or fault code."),
            sources=_sources(hits[:2]), mode="no_answer", grounded=False)

    with telemetry.span("rag.generate") as generate_span:
        llm = _llm_answer(query, hits)
        mode = llm[1] if llm is not None else "offline_extractive"
        # Which tier of the fallback actually served the answer. If this reads
        # "offline_extractive" in production when a key is configured, the LLM
        # path is silently failing -- exactly the kind of degradation that is
        # invisible without tracing, because the service still returns 200.
        telemetry.set_attributes(generate_span, **{"rag.mode": mode})

    if llm is not None:
        return RagAnswer(answer=llm[0], sources=_sources(hits), mode=mode, grounded=True)
    return RagAnswer(answer=_offline_answer(query, hits), sources=_sources(hits),
                     mode="offline_extractive", grounded=True)
