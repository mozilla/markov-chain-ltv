from __future__ import annotations
import collections
import copy
import csv
from datetime import datetime 
import logging
from typing import List
import typing
import uuid

from google.cloud import bigquery
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .node import Node


class SchemaException(Exception):
    def __init__(self, missing_fields: typing.Iterable[str]):
        super().__init__(f"Expected fields [{', '.join(missing_fields)}]")


class MarkovChain(object):
    def __init__(
        self,
        edges: List[(str, str, float)],
        return_value: List[(str, float)] = None,
        dtype=np.float32,
        query: str = None,
    ) -> None:
        self.query = query
        self.nodes = sorted(
            [Node(n) for n in set((s for edge in edges for s in edge[:2]))]
        )
        self._node_name_map = {n.name: n for n in self.nodes}
        self._node_index_map = {n.name: i for i, n in enumerate(self.nodes)}
        self.N = len(self.nodes)

        # Set edges
        for _from, to, p in edges:
            self._node_name_map[_from].add_edge(to, p)

        self._R = np.zeros(self.N, dtype=dtype)
        self._recalculate_R = True
        if return_value is not None:
            self.set_return_value(return_value)

        self._T = 1
        self._P = np.zeros(1, dtype=dtype)
        self._recalculate_P = True
        self._P_t = np.zeros(1, dtype=dtype)
        self._recalculate_P_t = True
        self._V = np.zeros(1, dtype=dtype)
        self._recalculate_V = True
        self._CV = np.zeros(1, dtype=dtype)
        self._recalculate_CV = True

    def clone(self) -> MarkovChain:
        """
        Create a clone of a markov chain.
        The new chain is completely independent
        of the source (i.e. they share no objects in memory).
        """
        return copy.deepcopy(self)

    @staticmethod
    def from_query(query: str, project=None, location=None) -> MarkovChain:
        """
        Create a markov chain from a BigQuery Query.

        @param query - The query that returns state transition probabilities.
            Queries must return rows of the form:
            [_from: str, _to: str, p: double]
        @param project (Optional) - The default project for BQ
        @param location (Optional) - The default location for BQ

        Project and location must be specified when using colab.
        """
        client = bigquery.Client(project=project, location=location)
        query_job = client.query(query)
        return MarkovChain.from_df(query_job.to_dataframe(), query)

    @staticmethod
    def from_csv(filename: str, ignore_header=True) -> MarkovChain:
        """
        Create a markov chain from a CSV file.

        @param filename - The name of the CSV file.
            Must start with three cols of (from: string, to: string, p: float)
            that represent state transition probabilities.
        @param ignore_header (default True) - Whether
            there is a header row in the CSV file. This function
            cares only about ordering (the first three cols are used).
        """
        edges = []
        with open(filename) as fd:
            reader = csv.reader(fd)
            for i, row in enumerate(reader):
                if ignore_header and i == 0:
                    continue
                edges.append((str(row[0]), str(row[1]), float(row[2])))

        return MarkovChain(edges)

    @staticmethod
    def from_df(df: pd.DataFrame, _query: str = None) -> MarkovChain:
        """
        Instatiate a Markov Chain from a Dataframe.

        @param df - Pandas dataframe, must have columns
            _from, _to, and p. Other columns are ignored.

        @param _query - Internal field, but you can pass in a query
            string if you used one to make the dataframe with this method
            directly rather than through from_query().
        """
        reqd_fields = set(("_from", "_to", "p"))
        found_fields = set(df.columns)

        # switch this to walrus when colab allows it
        missing_fields = reqd_fields - found_fields
        if missing_fields:
            raise SchemaException(missing_fields)

        edges = [(row["_from"], row["_to"], row["p"]) for _, row in df.iterrows()]
        return MarkovChain(edges, query=_query)

    """
    Node Functions

    When nodes change, the Chain will need to be recalculated
    We can:
      - Change transition probabilities (edge weights)
      - Reorder nodes

      TODO:
      - Add new nodes
      - Remove nodes
    """

    def get_edge(self, _from: str, to: str) -> float:
        return self._node_name_map[_from].get_p(to)

    def update_edge(self, _from: str, to: str, p: float) -> None:
        self._node_name_map[_from].update_edge(to, p)
        self._recalculate_P = True
        self._recalculate_P_t = True
        self._recalculate_V = True
        self._recalculate_CV = True

    def reorder(self, key_func: typing.Callable[str, int] = lambda x: x) -> None:
        self.nodes.sort(key=lambda n: key_func(n.name))
        self._recalculate_R = True
        self._recalculate_P = True
        self._recalculate_P_t = True
        self._recalculate_V = True
        self._recalculate_CV = True

    def validate(self) -> None:
        # Nodes are our source of truth,
        # if they are valid, the chain is valid
        all((n.validate() for n in self.nodes))

    def __getattr__(self, attr):
        if attr in ("T"):
            return self.get_T()
        if attr in ("R", "return_value"):
            return self.get_R()
        elif attr in ("P", "transition_matrix", "probabilities"):
            return self.get_P()
        elif attr in ("P_t", "future_probabilities"):
            return self.get_P_t()
        elif attr in ("V", "value"):
            return self.get_V()
        elif attr in ("CV", "cumulative_value"):
            return self.get_CV()
        elif attr in ("LTV", "lifetime_value"):
            return self.get_LTV()
        else:
            return super().__getattribute__(attr)

    def __setattr__(self, attr, value):
        # TODO: Reserve our faux attrs from __getattr__
        # e.g. if users set V, that could really screw things up
        if attr in ("T"):
            self.set_T(value)
        else:
            super().__setattr__(attr, value)

    def get_P(self) -> NDArray:
        if self._recalculate_P:
            self._calculate_P()
            self._recalculate_P = False
        return self._P

    def _calculate_P(self) -> None:
        self.validate()

        if self._P.shape != (self.N, self.N):
            self._P.resize((self.N, self.N), refcheck=False)
        for i, _from in enumerate(self.nodes):
            for j, to in enumerate(self.nodes):
                self._P[i][j] = _from.get_p(to.name)

    def set_T(self, T: int) -> None:
        assert T > 0, "T must be strictly positive (>0)"
        if T != self._T:
            self._T = T
            self._recalculate_P_t = True
            self._recalculate_V = True
            self._recalculate_CV = True

    def get_T(self) -> int:
        return self._T

    def get_P_t(self) -> NDArray:
        if self._recalculate_P_t:
            self._calculate_P_t()
            self._recalculate_P_t = False
        return self._P_t

    def _calculate_P_t(self) -> None:
        self.validate()

        if self._P_t.shape != (self._T, self.N, self.N):
            self._P_t.resize((self._T, self.N, self.N), refcheck=False)

        # We always include the 0th step (current time),
        # represented as the identity matrix
        self._P_t[0] = np.identity(self.N)
        for t in range(1, self._T):
            self._P_t[t] = self._P_t[t - 1] @ self.P

    def set_return_value(self, rv: Dict[str, float]) -> None:
        updated = False
        for node in self.nodes:
            if node.get_value() != rv.get(node.name, 0.0):
                updated = True
                node.set_value(rv.get(node.name, 0.0))
        if updated:
            self._recalculate_R = True
            self._recalculate_V = True
            self._recalculate_CV = True

    def get_R(self) -> NDArray:
        if self._recalculate_R:
            self._calculate_R()
            self._recalculate_R = False
        return self._R

    def _calculate_R(self) -> None:
        if self._R.shape != (self.N,):
            self._R.resize((self.N,), refcheck=False)
        for i, node in enumerate(self.nodes):
            self._R[i] = node.get_value()

    def get_V(self) -> NDArray:
        if self._recalculate_V:
            self._calculate_V()
            self._recalculate_V = False
        return self._V

    def _calculate_V(self) -> None:
        self.validate()

        if self._V.shape != (self._T, self.N):
            self._V.resize((self._T, self.N), refcheck=False)
        for t in range(self._T):
            self._V[t] = np.dot(self.P_t[t], self.R)

    def get_CV(self) -> NDArray:
        if self._recalculate_CV:
            self._calculate_CV()
            self._recalculate_CV = False
        return self._CV

    def _calculate_CV(self) -> None:
        self.validate()

        if self._CV.shape != (self._T, self.N):
            self._CV.resize((self._T, self.N), refcheck=False)
        for t in range(self._T):
            self._CV[t] = np.sum(np.transpose(self.V[: t + 1]), axis=1)

    def get_LTV(self, d=0.0) -> NDArray:
        return np.dot(
            np.linalg.inv(np.identity(self.N) - (self.P * (1 / (1 + d)))), self.R
        )

    """
    Prediction functions
    """

    def read_counts_from_query(
        self, query: str, project=None, location=None
    ) -> NDArray:
        """
        Retrieve state counts from a BigQuery query.

        @param query - The query to run.
            Queries must return rows of the form:
            [t: int, state: str, count: double]
            Other columns will be ignored
        @param project (Optional) - The default project for BQ
        @param location (Optional) - The default location for BQ
        """
        client = bigquery.Client(project=project, location=location)
        query_job = client.query(query)
        return MarkovChain.from_df(query_job.to_dataframe())

    def read_counts_from_csv(self, fname: str) -> NDArray:
        """
        Read state counts from a CSV file.
        Each row in the CSV is the counts for a single state
        at a single timestep, and is required to have columns
        [t, state, count] - and will error if any are missing.

        @param fname - The name of the csv file

        Returns an NDArray that the model can predict
        future counts from.
        """
        return self.read_counts_from_df(pd.read_csv(fname, dtype={"t": np.int64}))

    def read_counts_from_df(self, df: pd.DataFrame) -> NDArray:
        """
        Convert a DataFrame into an NDArray that can be used for prediction.

        @param df - Pandas dataframe, must have columns
            t, state, and count. Other columns are ignored.
        """
        reqd_fields = set(("t", "state", "count"))
        found_fields = set(df.columns)

        # switch this to walrus when colab allows it
        missing_fields = reqd_fields - found_fields
        if missing_fields:
            raise SchemaException(missing_fields)

        counts_map = collections.defaultdict(dict)

        for _, row in df.iterrows():
            counts_map[row["t"]][row["state"]] = row["count"]

        return np.array(
            [
                np.array([counts_map[t].get(node.name, 0) for node in self.nodes])
                for t in range(len(counts_map))
            ]
        )

    def predict_value(self, state_counts: NDArray, cumulative: bool = False) -> NDArray:
        """
        Given an array of state counts, predict the value
        of those states over the next t periods.

        @param state_counts - A (T, N) shape array, giving the number
            of additional counts for each state n<N at each future time
            step t<T.
        @param cumulative - Whether the result should be cumulative value.

        @return predictions - An NDArray of shape (T,) with the total value at each
            time step t<T. Value of states is determined by R, the return_value.
        """
        T, _ = state_counts.shape
        if T > self.T:
            self.T = T
        predictions = np.array(
            [
                np.sum(self.V[: t + 1] * np.flip(state_counts[: t + 1], axis=0))
                for t in range(T)
            ]
        )
        if cumulative:
            predictions = np.cumsum(predictions)
        return predictions

    def predict(self, state_counts: NDArray, cumulative: bool = False) -> NDArray:
        """
        Deprecated, use `predict_value` instead.
        """
        logging.warn(
            "`MarkovChain.predict` is deprecated. Use `MarkovChain.predict_value` instead"
        )
        return self.predict_value(state_counts, cumulative)

    def predict_states(self, state_counts: NDArray) -> NDArray:
        """
        Given an array of state counts, predict the counts
        of those states over the next t periods.

        @param state_counts - A (T, N) shape array, giving the number
            of additional counts for each state n<N at each future time
            step t<T.

        @return predictions - An NDArray of shape (T, N) with the predicted counts of each
            state n<N at each time step t<T.
        """
        T, _ = state_counts.shape
        if T > self.T:
            self.T = T

        totals = []
        for t in range(T):
            total = np.zeros(self.N)
            for p in range(t + 1):
                total += self.P_t[t - p].T @ state_counts[p]
            totals.append(total)

        return np.array(totals)


    """
    Saving and loading model query to BQ
    """

    def save_query_to_bq(self, username: str):
        """
        Save the model to BigQuery (telemetry_derived.markov_queries)

        @param username: your username, so it's clear who authored the query
        """
        assert (
            self.query is not None
        ), "You must have query data to save to use this function"
        execution_time = datetime.now().isoformat()
        output_dict = {
            "uuid": uuid.uuid4().hex,
            "author": username,
            "date_run": execution_time,
            "query": self.query,
        }
        output = pd.DataFrame(output_dict, index=[0])

        project = "mozdata"
        output_location = "analysis.core_insights_model_queries"

        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("uuid", "STRING"),
                bigquery.SchemaField("author", "STRING"),
                bigquery.SchemaField("date_run", "STRING"),
                bigquery.SchemaField("query", "STRING"),
            ]
        )
        client = bigquery.Client(project=project)
        client.load_table_from_dataframe(
            output, destination=output_location, job_config=job_config
        )

    @staticmethod
    def load_model_from_modeldb(
        query_id: str, project: str = None, location: str = None
    ):
        """
        Create a markov chain from a BigQuery Query stored in the model database.

        @param query_id - The id of the query you wish to use
        @param project (Optional) - The default project for BQ
        @param location (Optional) - The default location for BQ

        Project and location must be specified when using colab.
        """

        query = f"""select
        *
        from
        analysis.core_insights_model_queries
        where
        uuid = '{query_id}'"""

        bq_client = bigquery.Client(project="mozdata")
        results = bq_client.query(query).result().to_dataframe()
        query = results["query"][0]

        return MarkovChain.from_query(query, project=project, location=location)
