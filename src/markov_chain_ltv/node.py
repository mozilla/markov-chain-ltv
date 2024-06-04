from __future__ import annotations

from typing import Dict
from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray


class Node(object):
    def __init__(
        self, name: str, value: float = 0.0, edges: Dict[str, float] = None
    ) -> None:
        self.name = name
        if edges is None:
            edges = {}
        self.value = value
        self.edges = edges

    def add_edge(self, node: str, p: float) -> None:
        if node in self.edges.keys():
            raise Exception(
                f"Node {node} already present in edges of node {self}.\n\tEdges: {self.edges}"
            )
        self.edges[node] = p

    def update_edge(self, node: str, p: float) -> None:
        self.edges[node] = p

    def is_valid(self) -> bool:
        return abs(sum((v for _, v in self.edges.items())) - 1.0) < 10**-8

    def validate(self) -> None:
        if not self.is_valid():
            raise Exception(f"Invalid node {self.to_str()}")
        return True

    def get_p(self, node: str) -> float:
        return self.edges.get(node, 0.0)

    def get_value(self) -> float:
        return self.value

    def set_value(self, value: float) -> None:
        self.value = value

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def to_str(self) -> str:
        return f"Name: {self.name}, Edges: {self.edges.items()}"
