"""GNN model for DeadRecs — PyG data conversion and heterogeneous GraphSAGE.

Converts a NetworkX graph (with typed nodes/edges) to PyTorch Geometric
HeteroData, and defines a 2-layer heterogeneous GraphSAGE model for
self-supervised link prediction.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from deadrecs.features import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Era boundaries for one-hot encoding (6 eras)
ERA_BOUNDARIES = [
    ("pre_1970", None, "1970-01-01"),
    ("early_70s", "1970-01-01", "1974-01-01"),
    ("mid_70s", "1974-01-01", "1978-01-01"),
    ("late_70s", "1978-01-01", "1980-01-01"),
    ("80s", "1980-01-01", "1990-01-01"),
    ("90s", "1990-01-01", None),
]


def _era_onehot(date_str: str) -> list[float]:
    """Return a 6-dim one-hot vector for the show's era."""
    vec = [0.0] * len(ERA_BOUNDARIES)
    for i, (_, start, end) in enumerate(ERA_BOUNDARIES):
        after_start = start is None or date_str >= start
        before_end = end is None or date_str < end
        if after_start and before_end:
            vec[i] = 1.0
            break
    return vec


# ---------------------------------------------------------------------------
# NetworkX → PyG HeteroData conversion
# ---------------------------------------------------------------------------

# Edge type mapping from graph string types to PyG triplet format
EDGE_TYPE_MAP = {
    "HAS_PERFORMANCE": ("show", "has_performance", "performance"),
    "OF_SONG": ("performance", "of_song", "song"),
    "TRANSITIONED_TO": ("song", "transitioned_to", "song"),
    "SETLIST_NEIGHBOR": ("show", "setlist_neighbor", "show"),
}


def convert_to_heterodata(G: nx.DiGraph) -> tuple[HeteroData, dict[str, dict[str, int]]]:
    """Convert a NetworkX graph to PyTorch Geometric HeteroData.

    Returns:
        data: HeteroData with node features and edge indices.
        node_id_maps: dict mapping node type -> {nx_node_id: pyg_index}.
    """
    # Separate nodes by type and assign integer indices
    node_id_maps: dict[str, dict[str, int]] = defaultdict(dict)
    nodes_by_type: dict[str, list[str]] = defaultdict(list)

    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("type", "unknown")
        # Normalize type name for PyG
        if ntype == "performance":
            pyg_type = "performance"
        else:
            pyg_type = ntype
        idx = len(nodes_by_type[pyg_type])
        nodes_by_type[pyg_type].append(node_id)
        node_id_maps[pyg_type][node_id] = idx

    data = HeteroData()

    # Build node features
    _build_show_features(G, data, nodes_by_type.get("show", []))
    _build_song_features(G, data, nodes_by_type.get("song", []))
    _build_performance_features(G, data, nodes_by_type.get("performance", []))

    # Build edge indices
    edge_lists: dict[tuple, list[list[int]]] = defaultdict(lambda: [[], []])
    for u, v, attrs in G.edges(data=True):
        etype_str = attrs.get("type", "unknown")
        if etype_str not in EDGE_TYPE_MAP:
            continue

        pyg_etype = EDGE_TYPE_MAP[etype_str]
        src_type, _, dst_type = pyg_etype

        src_ntype = G.nodes[u].get("type", "unknown")
        dst_ntype = G.nodes[v].get("type", "unknown")

        # Normalize type names
        if src_ntype == "performance":
            src_ntype = "performance"
        if dst_ntype == "performance":
            dst_ntype = "performance"

        if src_ntype != src_type or dst_ntype != dst_type:
            continue

        src_idx = node_id_maps[src_type].get(u)
        dst_idx = node_id_maps[dst_type].get(v)

        if src_idx is not None and dst_idx is not None:
            edge_lists[pyg_etype][0].append(src_idx)
            edge_lists[pyg_etype][1].append(dst_idx)

    for etype, (src_list, dst_list) in edge_lists.items():
        if src_list:
            data[etype].edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )

    return data, dict(node_id_maps)


def _build_show_features(
    G: nx.DiGraph, data: HeteroData, show_nodes: list[str]
) -> None:
    """Build feature tensor for Show nodes.

    Features (8 dims): [num_performances, mean_vote, era_onehot (6 dims)]
    """
    if not show_nodes:
        data["show"].x = torch.zeros((0, 8))
        return

    features = []
    for node_id in show_nodes:
        attrs = G.nodes[node_id]
        date = attrs.get("date", "1970-01-01")

        # Count performances and compute mean vote
        perf_votes = []
        for _, neighbor, edata in G.out_edges(node_id, data=True):
            if edata.get("type") == "HAS_PERFORMANCE":
                v = G.nodes[neighbor].get("votes", 0)
                perf_votes.append(v)

        num_perfs = len(perf_votes)
        mean_vote = np.mean(perf_votes) if perf_votes else 0.0

        era = _era_onehot(date)
        features.append([float(num_perfs), float(mean_vote)] + era)

    data["show"].x = torch.tensor(features, dtype=torch.float)


def _build_song_features(
    G: nx.DiGraph, data: HeteroData, song_nodes: list[str]
) -> None:
    """Build feature tensor for Song nodes.

    Features (3 dims): [idf, total_votes, num_performances]
    """
    if not song_nodes:
        data["song"].x = torch.zeros((0, 3))
        return

    features = []
    for node_id in song_nodes:
        attrs = G.nodes[node_id]
        idf = attrs.get("idf", 0.0)

        # Count performances and total votes via incoming OF_SONG edges
        total_votes = 0
        num_perfs = 0
        for neighbor, _, edata in G.in_edges(node_id, data=True):
            if edata.get("type") == "OF_SONG":
                num_perfs += 1
                total_votes += G.nodes[neighbor].get("votes", 0)

        features.append([float(idf), float(total_votes), float(num_perfs)])

    data["song"].x = torch.tensor(features, dtype=torch.float)


def _build_performance_features(
    G: nx.DiGraph, data: HeteroData, perf_nodes: list[str]
) -> None:
    """Build feature tensor for SongPerformance nodes.

    Features (386 dims): [normalized_votes, rank_within_song, description_embedding (384 dims)]
    """
    if not perf_nodes:
        data["performance"].x = torch.zeros((0, 2 + EMBEDDING_DIM))
        return

    # Compute max votes for normalization
    all_votes = [G.nodes[n].get("votes", 0) for n in perf_nodes]
    max_votes = max(all_votes) if all_votes else 1
    if max_votes == 0:
        max_votes = 1

    # Compute rank within each song: group performances by song
    song_perfs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for node_id in perf_nodes:
        # Find the song this performance belongs to via OF_SONG edge
        for _, song_node, edata in G.out_edges(node_id, data=True):
            if edata.get("type") == "OF_SONG":
                votes = G.nodes[node_id].get("votes", 0)
                song_perfs[song_node].append((node_id, votes))
                break

    # Rank: sort by votes descending, rank 0 = best
    perf_ranks: dict[str, int] = {}
    for song_node, perfs in song_perfs.items():
        perfs.sort(key=lambda x: x[1], reverse=True)
        total = len(perfs)
        for rank, (perf_id, _) in enumerate(perfs):
            # Normalize rank to [0, 1]
            perf_ranks[perf_id] = rank / max(total - 1, 1)

    features = []
    for node_id in perf_nodes:
        votes = G.nodes[node_id].get("votes", 0)
        norm_votes = float(votes) / max_votes
        rank = perf_ranks.get(node_id, 0.0)

        embedding = G.nodes[node_id].get("embedding", None)
        if embedding is None:
            embedding = torch.zeros(EMBEDDING_DIM)
        elif not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float)

        feat = torch.cat([
            torch.tensor([norm_votes, rank], dtype=torch.float),
            embedding.float(),
        ])
        features.append(feat)

    data["performance"].x = torch.stack(features)


# ---------------------------------------------------------------------------
# Heterogeneous GraphSAGE Model
# ---------------------------------------------------------------------------


class DeadRecsGNN(nn.Module):
    """Heterogeneous GraphSAGE model for Grateful Dead show recommendations.

    2-layer GraphSAGE with HeteroConv, producing 128-dim node embeddings
    for all node types.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 128,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: per-type SAGEConv projecting to hidden_dim
        conv1_dict = {}
        for etype in metadata[1]:
            conv1_dict[etype] = SAGEConv((-1, -1), hidden_dim)
        self.conv1 = HeteroConv(conv1_dict, aggr="sum")

        # Layer 2: per-type SAGEConv projecting to out_dim
        conv2_dict = {}
        for etype in metadata[1]:
            conv2_dict[etype] = SAGEConv((-1, -1), out_dim)
        self.conv2 = HeteroConv(conv2_dict, aggr="sum")

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        """Forward pass returning embeddings for all node types."""
        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)

        return x_dict
