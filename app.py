"""Interface Streamlit. Toda a lógica fica no pacote `recomendador`."""

from __future__ import annotations

import streamlit as st

from recomendador.config import get_settings
from recomendador.db import MovieStore
from recomendador.embeddings import Embedder
from recomendador.llm import GeminiReranker, LlmError, Pick
from recomendador.recommender import Recommender
from recomendador.tmdb import Movie, TmdbClient, TmdbError

st.set_page_config(page_title="CineSense", page_icon="🎬", layout="wide")
settings = get_settings()


@st.cache_resource
def get_recommender() -> Recommender:
    tmdb = TmdbClient(settings.tmdb_token, settings.language, settings.region)
    store = MovieStore(settings.database_url, settings.embedding_dim)
    return Recommender(tmdb, store, Embedder(settings.embedding_model))


@st.cache_resource
def get_reranker() -> GeminiReranker | None:
    if not settings.gemini_api_key:
        return None
    return GeminiReranker(settings.gemini_api_key, settings.gemini_models)


st.title("🎬 CineSense")
st.caption("Recomendações que entendem clima, emoção e contexto, não só gênero e nota.")

try:
    rec = get_recommender()
except TmdbError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # banco fora do ar, etc.
    st.error(f"Não consegui iniciar: {e}. O banco está rodando (docker compose up -d)?")
    st.stop()

liked: dict[int, Movie] = st.session_state.setdefault("liked", {})

with st.sidebar:
    st.header("Seus filmes")
    if not liked:
        st.write("Nenhum filme escolhido ainda.")
    for mid, m in list(liked.items()):
        col1, col2 = st.columns([4, 1])
        col1.write(f"{m.title} ({m.year})")
        if col2.button("✕", key=f"rm{mid}"):
            del liked[mid]
            st.rerun()
    st.divider()
    discovery = st.slider("Descoberta", 0.0, 1.0, 0.3,
                          help="Quanto mais alto, mais filmes desconhecidos aparecem.")
    k = st.slider("Quantas sugestões", 5, 20, 10)
    reranker = get_reranker()
    use_llm = st.toggle("Refinar com IA (Gemini)", value=reranker is not None,
                        disabled=reranker is None,
                        help="O LLM reordena pelo clima e explica cada sugestão, sem spoilers.")
    if reranker is None:
        st.caption("Coloque GEMINI_API_KEY no .env para ativar.")

query = st.text_input("Buscar um filme que você gostou")
if query:
    for m in rec.tmdb.search(query)[:8]:
        c1, c2 = st.columns([1, 6])
        if m.poster_url:
            c1.image(m.poster_url, width=80)
        c2.markdown(f"**{m.title}** ({m.year}) · {', '.join(m.genres)}")
        c2.caption(m.overview[:220] + ("…" if len(m.overview) > 220 else ""))
        if c2.button("Gostei 👍", key=f"add{m.id}", disabled=m.id in liked):
            liked[m.id] = m
            st.rerun()

st.divider()
mood = st.text_input("Do que você está a fim? (opcional)",
                     placeholder="Ex.: algo tenso, mas sem gore exagerado",
                     disabled=not use_llm)

if st.button("Recomendar", type="primary", disabled=not liked):
    picks: list[Pick]
    with st.spinner("Buscando candidatos..."):
        # Com IA, buscamos 3x mais candidatos para o LLM escolher os melhores.
        pool = rec.recommend(list(liked), k=k * 3 if use_llm else k, discovery=discovery)
    picks = [Pick(c, None, None) for c in pool[:k]]
    if use_llm and reranker is not None:
        with st.spinner("A IA está analisando o clima de cada filme..."):
            try:
                picks = reranker.rerank(list(liked.values()), pool, k=k, mood=mood)
                st.caption(f"Refinado por {reranker.last_model} · {reranker.last_tokens} tokens")
            except LlmError as e:
                st.warning(f"A IA não respondeu agora; mostrando só a busca por similaridade. ({e})")

    for p in picks:
        m = p.candidate.movie
        c1, c2 = st.columns([1, 5])
        if m.poster_url:
            c1.image(m.poster_url, width=110)
        score = f"Nota da IA: {p.score:.0f}/10 · " if p.score is not None else ""
        c2.markdown(f"**{m.title}** ({m.year}) · {', '.join(m.genres)}")
        c2.caption(f"{score}Afinidade: {p.candidate.similarity:.0%} · ⭐ {m.vote_average:.1f}")
        if p.reason:
            c2.write(p.reason)

st.caption("Dados de filmes fornecidos pela TMDB. Este produto usa a API da TMDB, "
           "mas não é endossado nem certificado pela TMDB.")
