import pandas as pd
import numpy as np
from ..core.graph import StellarGraph, StellarDiGraph

class IAContact:
    def load(self):
        edges_path = '../../data/ia-contact.edges'
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class IARadoslawEmail:
    def load(self):
        edges_path = '../../data/ia-radoslaw-email.edges'
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class IAContactsHypertext2009:
    def load(self):
        edges_path = '../../data/ia-contacts_hypertext2009.edges'
        edges = pd.read_csv(
            edges_path,
            sep=r',',
            header=None,
            names=["source", "target", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges
