import pandas as pd
import pickle
import pytest
import numpy as np
import tempfile
import time
from markov_chain_ltv.chain import MarkovChain, SchemaException

@pytest.fixture
def simple_chain():
    return MarkovChain([("a", "b", 1), ("b", "a", 1)])

def test_node_ordering(simple_chain):
    assert [str(n) for n in simple_chain.nodes] == ["a", "b"]

def test_node_reordering(simple_chain):
    simple_chain.reorder(lambda x: 0 if x == "b" else 1)
    assert [str(n) for n in simple_chain.nodes] == ["b", "a"]

def test_P_calculation(simple_chain):
    assert simple_chain._recalculate_P
    assert (simple_chain.P == np.array([[0.0, 1.0], [1.0, 0.0]])).all()
    assert not simple_chain._recalculate_P

def test_P_recalculation(simple_chain):
    assert (simple_chain.P == np.array([[0.0, 1.0], [1.0, 0.0]])).all()
    simple_chain.update_edge("a", "b", 0)
    simple_chain.update_edge("a", "a", 1)
    assert simple_chain._recalculate_P

def test_negative_T(simple_chain):
    with pytest.raises(Exception):
        simple_chain.T = -1

def test_P_t_calculation(simple_chain):
    simple_chain.T = 2
    assert (simple_chain.P_t == np.array([[[1, 0], [0, 1]],[[0, 1], [1, 0]]])).all()

def test_P_t_recalculation(simple_chain):
    simple_chain.T = 2
    assert (simple_chain.P_t == np.array([[[1, 0], [0, 1]],[[0, 1], [1, 0]]])).all()
    simple_chain.T = 1
    assert simple_chain._recalculate_P_t
    assert (simple_chain.P_t == np.array([[[1, 0], [0, 1]]])).all()
    assert not simple_chain._recalculate_P_t

def test_return_value(simple_chain):
    simple_chain.set_return_value({"a": 1})
    assert (simple_chain.R == np.array([1, 0])).all()

def test_reorder_return_value(simple_chain):
    simple_chain.set_return_value({"a": 1})
    simple_chain.reorder(lambda x: 0 if x == "b" else 1)
    assert (simple_chain.R == np.array([0, 1])).all()

def test_V_calculation(simple_chain):
    simple_chain.set_return_value({"a": 1})
    simple_chain.T = 2
    assert (simple_chain.V == np.array([[1, 0], [0, 1]])).all()

def test_V_recalculation(simple_chain):
    simple_chain.set_return_value({"a": 1})
    simple_chain.T = 2
    assert simple_chain.V is not None
    simple_chain.set_return_value({"b": 1})
    assert (simple_chain.V == np.array([[0, 1], [1, 0]])).all()

def test_validate(simple_chain):
    simple_chain.validate()

def test_validate_false(simple_chain):
    simple_chain.update_edge("a", "b", 0.5)
    with pytest.raises(Exception):
        simple_chain.validate()

def test_clone(simple_chain):
    copy = simple_chain.clone()
    copy.update_edge("a", "a", 1)
    copy.update_edge("a", "b", 0)
    assert (copy.P != simple_chain.P).any()

def test_clone_b(simple_chain):
    # Try and guarantee that we're getting
    # completely independent objects after clone
    assert simple_chain.P is not None
    assert not simple_chain._recalculate_P
    copy = simple_chain.clone()
    assert copy.P is not None
    assert not copy._recalculate_P

    # This is only comparing values
    assert (copy.P == simple_chain.P).all()

    # ID still doesn't always guarantee distinct objects
    assert id(copy.P) != id(simple_chain.P)

    # Never actually do this, but see if setting
    # a single value in the copy alters the original
    # At this point, we're pretty sure that the np
    # arrays have been copied to new memory locations
    copy._P[0][0] = 27
    assert (copy.P != simple_chain.P).any()

def test_pickle(simple_chain):
    tmp_file = tempfile.NamedTemporaryFile()

    with open(tmp_file.name, 'wb') as f:
        pickle.dump(simple_chain, f)
    with open(tmp_file.name, 'rb') as f:
        o_chain = pickle.load(tmp_file)

    assert (simple_chain.P == o_chain.P).all()
    assert id(simple_chain.P) != id(o_chain.P)
    assert (simple_chain.R == o_chain.R).all()

def test_from_df(simple_chain):
    df = pd.DataFrame([
        {
            "_from": "a",
            "_to": "b",
            "p": 1,
        },
        {
            "_from": "b",
            "_to": "a",
            "p": 1,
        },
    ])
    df_chain = MarkovChain.from_df(df)

    assert (df_chain.P == simple_chain.P).all()

def test_from_improper_df(simple_chain):
    df = pd.DataFrame([
        {
            "wrong": "a",
            "_to": "b",
            "p": 1,
        },
        {
            "wrong": "b",
            "_to": "a",
            "p": 1,
        },
    ])

    with pytest.raises(SchemaException):
        df_chain = MarkovChain.from_df(df)

def test_read_counts_from_csv(simple_chain):
    df = pd.DataFrame([
        {"t": 1, "state": "a", "count": 4},
        {"t": 1, "state": "b", "count": 3},
        {"t": 0, "state": "a", "count": 2},
        {"t": 0, "state": "b", "count": 1},
    ])

    with tempfile.NamedTemporaryFile() as csv_file:
        df.to_csv(csv_file.name, index=False)
        counts = simple_chain.read_counts_from_csv(csv_file.name)
    assert (counts == np.array([[2, 1], [4, 3]])).all()

def test_read_counts_from_df(simple_chain):
    df = pd.DataFrame([
        {"t": 1, "state": "a", "count": 4},
        {"t": 1, "state": "b", "count": 3},
        {"t": 0, "state": "a", "count": 2},
        {"t": 0, "state": "b", "count": 1},
    ])

    counts = simple_chain.read_counts_from_df(df)
    assert (counts == np.array([[2, 1], [4, 3]])).all()

def test_predict(simple_chain):
    simple_chain.set_return_value({"b": 1})

    df = pd.DataFrame([
        {"t": 0, "state": "a", "count": 2},
        {"t": 0, "state": "b", "count": 1},
        {"t": 1, "state": "a", "count": 4},
        {"t": 2, "state": "a", "count": 6},
    ])

    counts = simple_chain.read_counts_from_df(df)
    prediction = simple_chain.predict_value(counts)

    assert (prediction == np.array([1, 2, 5])).all()

def test_predict_states(simple_chain):
    simple_chain.set_return_value({"b": 1})

    df = pd.DataFrame([
        {"t": 0, "state": "a", "count": 2},
        {"t": 0, "state": "b", "count": 1},
        {"t": 1, "state": "a", "count": 4},
        {"t": 2, "state": "a", "count": 6},
    ])

    counts = simple_chain.read_counts_from_df(df)
    prediction = simple_chain.predict_states(counts)

    assert (prediction == np.array([
                np.array([2, 1]),
                np.array([5, 2]),
                np.array([8, 5]),
            ])).all()
