"""Regressão: o pgvector devolve objetos Vector, não listas."""

import numpy as np

from recomendador.db import as_array


class FakeVector:
    def __init__(self, values):
        self._values = values

    def to_numpy(self):
        return np.array(self._values)


def test_converte_objeto_vector_do_pgvector():
    out = as_array(FakeVector([0.1, 0.2, 0.3]))
    assert out.dtype == np.float32 and np.allclose(out, [0.1, 0.2, 0.3])


def test_continua_aceitando_lista():
    assert np.allclose(as_array([1, 2]), [1.0, 2.0])
