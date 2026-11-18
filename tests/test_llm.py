from __future__ import annotations

import json

import pytest

from recomendador.db import Candidate
from recomendador.llm import GeminiReranker, LlmError, build_prompt, merge_picks
from recomendador.tmdb import Movie


def cand(mid, title="Filme", overview="sinopse"):
    return Candidate(Movie(id=mid, title=title, overview=overview, genres=["Terror"],
                           release_date="2020-01-01"), 0.9)


CANDS = [cand(1, "A"), cand(2, "B"), cand(3, "C")]


def test_merge_ordena_pela_nota_e_guarda_motivo():
    picks = merge_picks(CANDS, [
        {"id": 2, "nota": 9, "motivo": "Tensão claustrofóbica."},
        {"id": 1, "nota": 4, "motivo": "Só o tema parecido."},
        {"id": 3, "nota": 7, "motivo": "Criatura bem feita."},
    ], k=3)
    assert [p.candidate.movie.id for p in picks] == [2, 3, 1]
    assert picks[0].reason == "Tensão claustrofóbica."


def test_merge_ignora_id_inventado_repetido_e_malformado():
    picks = merge_picks(CANDS, [
        {"id": 999, "nota": 10, "motivo": "inventado"},
        {"id": 1, "nota": 8, "motivo": "ok"},
        {"id": 1, "nota": 2, "motivo": "repetido"},
        {"id": "x", "nota": 5},
    ], k=3)
    assert [p.candidate.movie.id for p in picks] == [1, 2, 3]
    assert picks[0].reason == "ok"
    # Os que o LLM não avaliou entram depois, sem nota.
    assert picks[1].score is None and picks[2].reason is None


def test_merge_limita_nota_e_respeita_k():
    picks = merge_picks(CANDS, [{"id": 3, "nota": 42, "motivo": "m"}], k=1)
    assert len(picks) == 1 and picks[0].score == 10.0


def test_prompt_inclui_pedido_e_corta_sinopse_longa():
    longo = cand(5, "Longo", "x" * 1000)
    p = build_prompt([Movie(id=9, title="Alien", overview="nave")], [longo], "algo tenso", k=10)
    assert "algo tenso" in p and "Alien" in p and "5 | Longo" in p
    assert "Escolha os 1 candidatos" in p  # nunca pede mais do que existe
    assert "x" * 401 not in p


class FakeModels:
    def __init__(self, behavior):
        self.behavior = behavior  # modelo -> resposta (str) ou exceção
        self.called = []

    def generate_content(self, model, contents, config):
        self.called.append(model)
        result = self.behavior[model]
        if isinstance(result, Exception):
            raise result

        class Resp:
            text = result
            usage_metadata = type("U", (), {"total_token_count": 123})()

        return Resp()


class FakeClient:
    def __init__(self, behavior):
        self.models = FakeModels(behavior)


def test_usa_modelo_reserva_quando_o_primeiro_falha():
    ok = json.dumps({"recomendacoes": [{"id": 3, "nota": 9, "motivo": "boa"}]})
    client = FakeClient({"rapido": RuntimeError("503 UNAVAILABLE"), "reserva": ok})
    r = GeminiReranker("", ["rapido", "reserva"], client=client)
    picks = r.rerank([CANDS[0].movie], CANDS, k=2)
    assert client.models.called == ["rapido", "reserva"]
    assert r.last_model == "reserva" and r.last_tokens == 123
    assert picks[0].candidate.movie.id == 3


def test_erro_claro_quando_todos_falham():
    client = FakeClient({"a": RuntimeError("503"), "b": "isto não é json"})
    with pytest.raises(LlmError):
        GeminiReranker("", ["a", "b"], client=client).rerank([], CANDS, k=2)


def test_sem_chave_gera_erro():
    with pytest.raises(LlmError):
        GeminiReranker("", ["a"])
