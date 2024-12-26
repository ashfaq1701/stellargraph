import argparse
import pickle
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from temporal_walk import TemporalWalk
from sklearn.model_selection import train_test_split
import random
from stellargraph import StellarGraph
from stellargraph.data import TemporalRandomWalk, BiasedRandomWalk
from stellargraph.datasets import FBForum, IAContact, IAContactsHypertext2009, IAEmailEU, IAEnronEmployees, \
    IARadoslawEmail, SocSignBitcoinAlpha, WikiElections
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TRAIN_SUBSET = 0.25
TEST_SUBSET = 0.25
EMBEDDING_SIZE = 128

NUM_WALKS_PER_NODE = 10
WALK_LENGTH = 80
CONTEXT_WINDOW_SIZE = 10


def get_dataset(dataset_name):
    if dataset_name == 'fb_forum':
        return FBForum()
    elif dataset_name == 'ia_contact':
        return IAContact()
    elif dataset_name == 'ia_contacts_hypertext_2009':
        return IAContactsHypertext2009()
    elif dataset_name == 'ia_email_eu':
        return IAEmailEU()
    elif dataset_name == 'ia_enron_employees':
        return IAEnronEmployees()
    elif dataset_name == 'ia_radoslaw_email':
        return IARadoslawEmail()
    elif dataset_name == 'soc_sign_bitcoin_alpha':
        return SocSignBitcoinAlpha()
    elif dataset_name == 'wiki_elections':
        return WikiElections()
    else:
        raise ValueError(f'Invalid dataset name {dataset_name}')


def positive_and_negative_links(g, edges):
    pos = list(edges[["source", "target"]].itertuples(index=False))
    neg = sample_negative_examples(g, pos)
    return pos, neg


def sample_negative_examples(g, positive_examples):
    positive_set = set(positive_examples)

    def valid_neg_edge(src, tgt):
        return (
            # no self-loops
            src != tgt
            and
            # neither direction of the edge should be a positive one
            (src, tgt) not in positive_set
            and (tgt, src) not in positive_set
        )

    possible_neg_edges = [
        (src, tgt) for src in g.nodes() for tgt in g.nodes() if valid_neg_edge(src, tgt)
    ]
    return random.sample(possible_neg_edges, k=len(positive_examples))


def compute_link_prediction_auc(dataset_name, walk_bias, initial_edge_bias):
    dataset = get_dataset(dataset_name)
    full_graph, edges = dataset.load()
    edges_list = [(int(row[0]), int(row[1]), row[2]) for row in edges.to_numpy()]

    num_edges_graph = int(len(edges) * (1 - TRAIN_SUBSET))

    edges_graph = edges[:num_edges_graph]
    edges_other = edges[num_edges_graph:]

    edges_train, edges_test = train_test_split(edges_other, test_size=TEST_SUBSET)

    graph = StellarGraph(
        nodes=pd.DataFrame(index=full_graph.nodes()),
        edges=edges_graph,
        edge_weight_column="time",
    )

    pos, neg = positive_and_negative_links(graph, edges_train)
    pos_test, neg_test = positive_and_negative_links(graph, edges_test)

    num_cw = len(graph.nodes()) * NUM_WALKS_PER_NODE * (WALK_LENGTH - CONTEXT_WINDOW_SIZE + 1)

    temporal_walk_old_start_time = time.time()
    temporal_rw = TemporalRandomWalk(graph)
    temporal_walks_old = temporal_rw.run(
        num_cw=num_cw,
        cw_size=CONTEXT_WINDOW_SIZE,
        max_walk_length=WALK_LENGTH,
        walk_bias=walk_bias.lower(),
    )
    temporal_walk_old_time = time.time() - temporal_walk_old_start_time

    temporal_walk_new_start_time = time.time()
    temporal_walk = TemporalWalk()
    temporal_walk.add_multiple_edges(edges_list)
    temporal_walks = temporal_walk.get_random_walks(
        max_walk_len=WALK_LENGTH,
        walk_bias=walk_bias,
        num_cw=num_cw,
        initial_edge_bias=initial_edge_bias,
        walk_direction="Forward_In_Time",
        walk_init_edge_time_bias="Bias_Earliest_Time",
        context_window_len=CONTEXT_WINDOW_SIZE
    )
    temporal_walks_new = [[str(node) for node in walk] for walk in temporal_walks]
    temporal_walk_new_time = time.time() - temporal_walk_new_start_time

    static_rw = BiasedRandomWalk(graph)
    static_walks = static_rw.run(
        nodes=graph.nodes(), n=NUM_WALKS_PER_NODE, length=WALK_LENGTH
    )

    temporal_model_old = Word2Vec(
        temporal_walks_old,
        vector_size=EMBEDDING_SIZE,
        window=CONTEXT_WINDOW_SIZE,
        min_count=0,
        sg=1,
        workers=2,
        epochs=1,
    )
    temporal_model_new = Word2Vec(
        temporal_walks_new,
        vector_size=EMBEDDING_SIZE,
        window=CONTEXT_WINDOW_SIZE,
        min_count=0,
        sg=1,
        workers=2,
        epochs=1,
    )

    static_model = Word2Vec(
        static_walks,
        vector_size=EMBEDDING_SIZE,
        window=CONTEXT_WINDOW_SIZE,
        min_count=0,
        sg=1,
        workers=2,
        epochs=1,
    )

    unseen_node_embedding = np.zeros(EMBEDDING_SIZE)

    def temporal_embedding(is_new):
        def get_for_node(u):
            try:
                if is_new:
                    return temporal_model_new.wv[u]
                else:
                    return temporal_model_old.wv[u]
            except KeyError:
                return unseen_node_embedding
        return get_for_node

    def static_embedding(u):
        return static_model.wv[u]

    def operator_l2(u, v):
        return (u - v) ** 2

    binary_operator = operator_l2

    def link_examples_to_features(link_examples, transform_node):
        op_func = (
            operator_func[binary_operator]
            if isinstance(binary_operator, str)
            else binary_operator
        )
        return [
            op_func(transform_node(src), transform_node(dst)) for src, dst in link_examples
        ]

    def link_prediction_classifier(max_iter=2000):
        lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
        return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

    def evaluate_roc_auc(clf, link_features, link_labels):
        predicted = clf.predict_proba(link_features)

        # check which class corresponds to positive links
        positive_column = list(clf.classes_).index(1)
        return roc_auc_score(link_labels, predicted[:, positive_column])

    def labelled_links(positive_examples, negative_examples):
        return (
            positive_examples + negative_examples,
            np.repeat([1, 0], [len(positive_examples), len(negative_examples)]),
        )

    link_examples, link_labels = labelled_links(pos, neg)
    link_examples_test, link_labels_test = labelled_links(pos_test, neg_test)

    temporal_link_features_old = link_examples_to_features(link_examples, temporal_embedding(False))
    temporal_link_features_test_old = link_examples_to_features(link_examples_test, temporal_embedding(False))
    temporal_clf_old = link_prediction_classifier()
    temporal_clf_old.fit(temporal_link_features_old, link_labels)
    temporal_score_old = evaluate_roc_auc(
        temporal_clf_old, temporal_link_features_test_old, link_labels_test
    )
    print(f"Score Temporal - Old: {temporal_score_old:.2f}")

    temporal_link_features_new = link_examples_to_features(link_examples, temporal_embedding(True))
    temporal_link_features_test_new = link_examples_to_features(link_examples_test, temporal_embedding(True))
    temporal_clf_new = link_prediction_classifier()
    temporal_clf_new.fit(temporal_link_features_new, link_labels)
    temporal_score_new = evaluate_roc_auc(
        temporal_clf_new, temporal_link_features_test_new, link_labels_test
    )
    print(f"Score Temporal - New: {temporal_score_new:.2f}")

    static_clf = link_prediction_classifier()
    static_link_features = link_examples_to_features(link_examples, static_embedding)
    static_link_features_test = link_examples_to_features(
        link_examples_test, static_embedding
    )
    static_clf.fit(static_link_features, link_labels)
    static_score = evaluate_roc_auc(static_clf, static_link_features_test, link_labels_test)
    print(f"Score Static: {static_score:.2f}")

    return {
        'auc_static': static_score,
        'auc_temporal_old': temporal_score_old,
        'auc_temporal_new': temporal_score_new,
        'temporal_walk_old_time': temporal_walk_old_time,
        'temporal_walk_new_time': temporal_walk_new_time
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal link prediction.")

    # Add arguments to the parser
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--walk_bias', type=str, default="Exponential")
    parser.add_argument('--initial_edge_bias', type=str, default="Uniform")
    parser.add_argument('--n_runs', type=int, default=5)

    args = parser.parse_args()

    auc_statics = []
    auc_temporal_olds = []
    auc_temporal_news = []
    temporal_walk_old_times = []
    temporal_walk_new_times = []

    for _ in range(args.n_runs):
        result = compute_link_prediction_auc(args.dataset, args.walk_bias, args.initial_edge_bias)

        auc_statics.append(result["auc_static"])
        auc_temporal_olds.append(result["auc_temporal_old"])
        auc_temporal_news.append(result["auc_temporal_new"])
        temporal_walk_old_times.append(result["temporal_walk_old_time"])
        temporal_walk_new_times.append(result["temporal_walk_new_time"])

    combined_result = {
        'auc_static': auc_statics,
        'auc_temporal_old': auc_temporal_olds,
        'auc_temporal_new': auc_temporal_news,
        'temporal_walk_old_time': temporal_walk_old_times,
        'temporal_walk_new_time': temporal_walk_new_times
    }

    pickle.dump(combined_result, open(f"save/{args.dataset}_{args.walk_bias}_{args.initial_edge_bias}.pkl", "wb"))

    print(f"Auc Static: {np.mean(auc_statics)}")
    print(f"Auc Temporal (Old): {np.mean(auc_temporal_olds)}")
    print(f"Auc Temporal (New): {np.mean(auc_temporal_news)}")
    print(f"Temporal walk sampling time (Old): {np.mean(temporal_walk_old_times)}")
    print(f"Temporal walk sampling time (New): {np.mean(temporal_walk_new_times)}")
