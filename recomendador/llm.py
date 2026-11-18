"""Etapa de LLM: reordena candidatos pelo clima/emoção e explica cada escolha."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .db import Candidate
from .tmdb import Movie

SYSTEM_PROMPT = """Você é um curador de cinema que recomenda filmes para uma pessoa específica.
Regras:
- Avalie o CLIMA, o TOM, a EMOÇÃO e o RITMO de cada candidato, não só o gênero.
  Filmes que apenas compartilham palavras ou tema superficial com os favoritos devem
  receber nota baixa.
- NUNCA revele spoilers: reviravoltas, final, quem morre ou segredos da trama.
- Para cada candidato, dê uma nota de 0 a 10 de quanto ele combina com a pessoa e
  um motivo curto (1 ou 2 frases, em português do Brasil) ligando o filme ao gosto
  dela. Fale diretamente com a pessoa ("você").
- Use apenas os ids da lista de candidatos."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "recomendacoes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "nota": {"type": "number"},
                    "motivo": {"type": "string"},
                },
                "required": ["id", "nota", "motivo"],
            },
        }
    },
    "required": ["recomendacoes"],
}

MAX_OVERVIEW = 400  # limita tokens (e custo) por filme


class LlmError(RuntimeError):
    pass


@dataclass
class Pick:
    candidate: Candidate
    score: float | None  # None = o LLM não avaliou; ficou pela ordem dos embeddings
    reason: str | None


def _describe(m: Movie) -> str:
    overview = m.overview[:MAX_OVERVIEW] + ("…" if len(m.overview) > MAX_OVERVIEW else "")
    return f"{m.title} ({m.year}) | {', '.join(m.genres)} | {overview}"


def build_prompt(
    liked: list[Movie], candidates: list[Candidate], mood: str | None, k: int
) -> str:
    lines = [
        f"Escolha os {min(k, len(candidates))} candidatos que mais combinam com a pessoa "
        "e devolva exatamente essa quantidade, do melhor para o pior.",
        "",
        "Filmes que a pessoa gostou:",
    ]
    lines += [f"- {_describe(m)}" for m in liked]
    if mood and mood.strip():
        lines += ["", f"O que a pessoa pediu agora: {mood.strip()}"]
    lines += ["", "Candidatos (id | título | gêneros | sinopse):"]
    lines += [f"- {c.movie.id} | {_describe(c.movie)}" for c in candidates]
    return "\n".join(lines)


def merge_picks(candidates: list[Candidate], items: list[dict[str, Any]], k: int) -> list[Pick]:
    """Junta a resposta do LLM com os candidatos, de forma defensiva.

    Ignora ids inventados ou repetidos, limita a nota a 0–10 e, se o LLM
    deixar candidatos de fora, completa com eles na ordem original.
    """
    by_id = {c.movie.id: c for c in candidates}
    picks: list[Pick] = []
    seen: set[int] = set()
    for item in items:
        try:
            mid = int(item["id"])
            score = min(max(float(item["nota"]), 0.0), 10.0)
        except (KeyError, TypeError, ValueError):
            continue
        if mid in by_id and mid not in seen:
            seen.add(mid)
            picks.append(Pick(by_id[mid], score, str(item.get("motivo") or "").strip() or None))
    picks.sort(key=lambda p: p.score, reverse=True)
    picks += [Pick(c, None, None) for c in candidates if c.movie.id not in seen]
    return picks[:k]


class GeminiReranker:
    """Tenta os modelos em ordem: se um estiver sobrecarregado, usa o próximo."""

    def __init__(self, api_key: str, models: list[str], client: Any = None) -> None:
        if not api_key and client is None:
            raise LlmError("GEMINI_API_KEY não configurada.")
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self.models = models
        self.last_model: str | None = None
        self.last_tokens: int | None = None

    def _generate(self, prompt: str) -> dict[str, Any]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.4,
        )
        errors = []
        for model in self.models:
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                data = json.loads(resp.text)
                self.last_model = model
                usage = getattr(resp, "usage_metadata", None)
                self.last_tokens = getattr(usage, "total_token_count", None)
                return data
            except Exception as e:  # 503/429/JSON inválido: tenta o próximo modelo
                errors.append(f"{model}: {str(e)[:120]}")
        raise LlmError("Nenhum modelo respondeu. " + " | ".join(errors))

    def rerank(
        self, liked: list[Movie], candidates: list[Candidate], k: int, mood: str | None = None
    ) -> list[Pick]:
        if not candidates:
            return []
        data = self._generate(build_prompt(liked, candidates, mood, k))
        return merge_picks(candidates, data.get("recomendacoes", []), k)
