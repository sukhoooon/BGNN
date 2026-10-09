"""One-way conversion of explicitly trusted local graph pickles.

Pickle can execute arbitrary code. This adapter is NOT a security sandbox.
Never use it on downloaded or otherwise untrusted objects.
"""
import argparse
import pickle
from pathlib import Path

from .data import save_dataset
from .features import UnaryMolData


class GraphUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("feature", "script.feature") and name == "UnaryMolData":
            return UnaryMolData
        return super().find_class(module, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trust-local-pickle", action="store_true")
    args = parser.parse_args(argv)
    if not args.trust_local_pickle:
        parser.error("Explicit --trust-local-pickle acknowledgement is required")
    if args.output.exists():
        parser.error("Output exists")
    with args.source.open("rb") as stream:
        graphs = GraphUnpickler(stream).load()
    if not isinstance(graphs, list) or not all(isinstance(g, UnaryMolData) for g in graphs):
        raise ValueError("Expected a unary preprocessed graph list")
    for graph in graphs:
        if graph.edge_index.numel() == 0:
            graph.edge_index = graph.edge_index.reshape(2, 0)
            graph.edge_attr = graph.edge_attr.reshape(0, 7)
    save_dataset(graphs, args.output)
    print(f"Converted {len(graphs)} graphs; original tensors and targets retained")


if __name__ == "__main__":
    main()
