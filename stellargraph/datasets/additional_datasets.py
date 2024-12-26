from io import StringIO
import pandas as pd
import numpy as np
from ..core.graph import StellarGraph, StellarDiGraph
import os


def resolve_path(*possible_paths):
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the specified paths exist: {possible_paths}")


class IAContact:
    def load(self):
        edges_path = resolve_path(
            '../../data/ia-contact.edges',
            '../../../data/ia-contact.edges',
            '../../../../data/ia-contact.edges'
        )
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
        edges_path = resolve_path(
            '../../data/ia-radoslaw-email.edges',
            '../../../data/ia-radoslaw-email.edges',
            '../../../../data/ia-radoslaw-email.edges'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python',
            skiprows=2
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
        edges_path = resolve_path(
            '../../data/ia-contacts_hypertext2009.edges',
            '../../../data/ia-contacts_hypertext2009.edges',
            '../../../../data/ia-contacts_hypertext2009.edges'
        )
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


class FBForum:
    def load(self):
        edges_path = resolve_path(
            '../../data/fb-forum.edges',
            '../../../data/fb-forum.edges',
            '../../../../data/fb-forum.edges'
        )
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


class SocSignBitcoinAlpha:
    def load(self):
        edges_path = resolve_path(
            '../../data/out.soc-sign-bitcoinalpha',
            '../../../data/out.soc-sign-bitcoinalpha',
            '../../../../data/out.soc-sign-bitcoinalpha'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            skiprows=1,
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


class IAEmailEU:
    def load(self):
        edges_path = resolve_path(
            '../../data/email-Eu-core-temporal.txt',
            '../../../data/email-Eu-core-temporal.txt',
            '../../../../data/email-Eu-core-temporal.txt',
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
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


class WikiElections:
    def load_from_string(self, data_string):
        sources = []
        targets = []
        times = []
        votes = []

        for line in StringIO(data_string):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            if parts[0] == 'E':
                continue
            elif parts[0] == 'U':
                candidate_id = parts[1]
            elif parts[0] == 'T':
                election_time = pd.to_datetime(f"{parts[1]} {parts[2]}")
            elif parts[0] == 'N':
                nominator_id = parts[1]
                sources.append(nominator_id)
                targets.append(candidate_id)
                times.append(election_time)
                votes.append(1)
            elif parts[0] == 'V':
                vote_value = int(parts[1])
                voter_id = parts[2]
                vote_time = f"{parts[3]} {parts[4]}"

                sources.append(voter_id)
                targets.append(candidate_id)
                times.append(pd.to_datetime(vote_time))
                votes.append(vote_value)

        edges = pd.DataFrame({
            'source': sources,
            'target': targets,
            'timestamp': times,
            'vote': votes
        })

        edges['timestamp'] = pd.to_numeric(edges['timestamp'].fillna(
            edges['timestamp'].min()
        ).astype('int64')) // 10 ** 9

        edges[['source', 'target']] = edges[['source', 'target']].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges['source'], edges['target']], ignore_index=True)
            )
        )

        graph = StellarGraph(nodes=nodes, edges=edges, edge_weight_column='timestamp')

        return graph, edges

    def load(self):
        file_path = resolve_path(
            '../../data/wikiElec.ElecBs3.txt',
            '../../../data/wikiElec.ElecBs3.txt',
            '../../../../data/wikiElec.ElecBs3.txt'
        )
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = f.read()

        return self.load_from_string(data)