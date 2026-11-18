"""Interface Streamlit. Toda a lógica fica no pacote `recomendador`."""

from __future__ import annotations

import streamlit as st

from recomendador.config import get_settings
from recomendador.db import MovieStore
from recomendador.embeddings import Embedder
from recomendador.recommender import Recommender
from recomendador.tmdb import Movie, TmdbClient, TmdbError

st.set_page_config(page_title="Recomendador de Filmes com IA", page_icon="🎬", layout="wide")
settings = get_settings()


@st.cache_resource
def get_recommender() -> Recommender:
    tmdb = TmdbClient(settings.tmdb_token, settings.language, settings.region)
    store = MovieStore(settings.database_url, settings.embedding_dim)
    return Recommender(tmdb, store, Embedder(settings.embedding_model))


st.title("🎬 Recomendador de Filmes com IA")
st.caption("Escolha filmes que você gostou e receba sugestões que vão além do gênero.")

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
if st.button("Recomendar", type="primary", disabled=not liked):
    with st.spinner("Analisando seus gostos..."):
        results = rec.recommend(list(liked), k=k, discovery=discovery)
    cols = st.columns(5)
    for i, cand in enumerate(results):
        with cols[i % 5]:
            if cand.movie.poster_url:
                st.image(cand.movie.poster_url)
            st.markdown(f"**{cand.movie.title}** ({cand.movie.year})")
            st.caption(f"Afinidade: {cand.similarity:.0%} · ⭐ {cand.movie.vote_average:.1f}")

st.caption("Dados de filmes fornecidos pela TMDB. Este produto usa a API da TMDB, "
           "mas não é endossado nem certificado pela TMDB.")
