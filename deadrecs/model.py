"""PyTorch Geometric data conversion and GNN model definition.

Converts the NetworkX heterogeneous graph into a PyG HeteroData object
and defines a heterogeneous GraphSAGE model for link prediction.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import click
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from deadrecs.features import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Era boundaries for one-hot encoding (6 eras).
# Primal (pre-1966), Classic (1966-1974), Hiatus (1975-1976),
# Comeback (1976-1979), Eighties (1980-1989), Final (1990+)
ERA_BOUNDARIES = [
    (None, "1966-01-01"),
    ("1966-01-01", "1975-01-01"),
    ("1975-01-01", "1977-01-01"),
    ("1977-01-01", "1980-01-01"),
    ("1980-01-01", "1990-01-01"),
    ("1990-01-01", None),
]

SHOW_FEAT_DIM = 8  # num_performances + mean_vote + 6 era dims
SONG_FEAT_DIM = 3  # idf + total_votes + num_performances
PERF_FEAT_DIM = 2 + EMBEDDING_DIM  # normalized_votes + rank_within_song + 384


def _date_to_era_onehot(date_str: str) -> list[float]:
    """Convert a date string to a 6-dim one-hot era vector."""
    vec = [0.0] * 6
    for i, (start, end) in enumerate(ERA_BOUNDARIES):
        if start is not None and date_str < start:
            continue
        if end is not None and date_str >= end:
            continue
        vec[i] = 1.0
        break
    return vec


def convert_to_hetero_data(G: nx.DiGraph) -> tuple[HeteroData, dict[str, dict[str, int]]]:
    """Convert a NetworkX graph to a PyTorch Geometric HeteroData object.

    Returns:
        data: HeteroData with node features and edge indices.
        node_id_maps: Dict mapping node type -> {nx_node_id: pyg_index}.
    """
    click.echo("Converting graph to PyG HeteroData...")

    # Separate nodes by type and assign integer indices.
    node_id_maps: dict[str, dict[str, int]] = {
        "show": {},
        "song": {},
        "performance": {},
    }

    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("type")
        if ntype in node_id_maps:
            node_id_maps[ntype][node_id] = len(node_id_maps[ntype])

    # --- Build node features ---

    # Show features: [num_performances, mean_vote, era_onehot(6)]
    num_shows = len(node_id_maps["show"])
    show_features = torch.zeros(num_shows, SHOW_FEAT_DIM)
    for nx_id, idx in node_id_maps["show"].items():
        attrs = G.nodes[nx_id]
        # Count performances and mean votes via HAS_PERFORMANCE edges
        perf_votes = []
        for _, target, edata in G.out_edges(nx_id, data=True):
            if edata.get("type") == "HAS_PERFORMANCE":
                perf_votes.append(G.nodes[target].get("votes", 0))
        num_perfs = len(perf_votes)
        mean_vote = sum(perf_votes) / num_perfs if num_perfs > 0 else 0.0
        era = _date_to_era_onehot(attrs.get("date", "1970-01-01"))
        show_features[idx] = torch.tensor(
            [float(num_perfs), mean_vote] + era
        )

    # Song features: [idf, total_votes, num_performances]
    num_songs = len(node_id_maps["song"])
    song_features = torch.zeros(num_songs, SONG_FEAT_DIM)
    for nx_id, idx in node_id_maps["song"].items():
        attrs = G.nodes[nx_id]
        idf = attrs.get("idf", 0.0)
        # Aggregate votes from performance nodes via incoming OF_SONG edges
        total_votes = 0
        num_perfs = 0
        for source, _, edata in G.in_edges(nx_id, data=True):
            if edata.get("type") == "OF_SONG":
                total_votes += G.nodes[source].get("votes", 0)
                num_perfs += 1
        song_features[idx] = torch.tensor(
            [idf, float(total_votes), float(num_perfs)]
        )

    # SongPerformance features: [normalized_votes, rank_within_song, embedding(384)]
    num_perfs = len(node_id_maps["performance"])
    perf_features = torch.zeros(num_perfs, PERF_FEAT_DIM)

    # Compute rank within song: group performances by song, rank by votes descending.
    song_perfs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for nx_id, idx in node_id_maps["performance"].items():
        votes = G.nodes[nx_id].get("votes", 0)
        # Find the song this performance belongs to
        for _, target, edata in G.out_edges(nx_id, data=True):
            if edata.get("type") == "OF_SONG":
                song_perfs[target].append((nx_id, votes))
                break

    # Compute ranks (0-indexed, higher votes = lower rank number)
    perf_ranks: dict[str, int] = {}
    for song_id, perfs in song_perfs.items():
        sorted_perfs = sorted(perfs, key=lambda x: x[1], reverse=True)
        for rank, (perf_id, _) in enumerate(sorted_perfs):
            perf_ranks[perf_id] = rank

    # Find max votes for normalization
    all_votes = [G.nodes[nx_id].get("votes", 0) for nx_id in node_id_maps["performance"]]
    max_votes = max(all_votes) if all_votes else 1

    for nx_id, idx in node_id_maps["performance"].items():
        attrs = G.nodes[nx_id]
        votes = attrs.get("votes", 0)
        normalized_votes = votes / max_votes if max_votes > 0 else 0.0
        rank = float(perf_ranks.get(nx_id, 0))
        embedding = attrs.get("embedding", torch.zeros(EMBEDDING_DIM))
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        perf_features[idx, 0] = normalized_votes
        perf_features[idx, 1] = rank
        perf_features[idx, 2:] = embedding

    # Apply LayerNorm to performance features.
    # The 384-dim text embedding values cluster near zero, so without
    # normalization the votes and rank features would be drowned out.
    layer_norm = nn.LayerNorm(PERF_FEAT_DIM)
    with torch.no_grad():
        perf_features = layer_norm(perf_features)

    # --- Build HeteroData ---
    data = HeteroData()
    data["show"].x = show_features
    data["song"].x = song_features
    data["performance"].x = perf_features

    data["show"].num_nodes = num_shows
    data["song"].num_nodes = num_songs
    data["performance"].num_nodes = num_perfs

    # --- Build edge indices ---
    # Forward edge types from the NetworkX graph.
    edge_type_map = {
        "HAS_PERFORMANCE": ("show", "has_performance", "performance"),
        "OF_SONG": ("performance", "of_song", "song"),
        "TRANSITIONED_TO": ("song", "transitioned_to", "song"),
        "SETLIST_NEIGHBOR": ("show", "setlist_neighbor", "show"),
    }

    # Reverse edge types so information flows both directions.
    reverse_edge_types = {
        ("show", "has_performance", "performance"): ("performance", "rev_has_performance", "show"),
        ("performance", "of_song", "song"): ("song", "rev_of_song", "performance"),
        ("song", "transitioned_to", "song"): ("song", "rev_transitioned_to", "song"),
    }
    # SETLIST_NEIGHBOR is already show↔show; no separate reverse needed.

    all_edge_types = list(edge_type_map.values()) + list(reverse_edge_types.values())
    edge_indices: dict[tuple, list[list[int]]] = {
        et: [[], []] for et in all_edge_types
    }

    for src, dst, edata in G.edges(data=True):
        etype = edata.get("type")
        if etype not in edge_type_map:
            continue
        pyg_etype = edge_type_map[etype]
        src_type = pyg_etype[0]
        dst_type = pyg_etype[2]

        src_idx = node_id_maps[src_type].get(src)
        dst_idx = node_id_maps[dst_type].get(dst)

        if src_idx is not None and dst_idx is not None:
            # Forward edge
            edge_indices[pyg_etype][0].append(src_idx)
            edge_indices[pyg_etype][1].append(dst_idx)

            # Reverse edge (if one exists for this type)
            rev_etype = reverse_edge_types.get(pyg_etype)
            if rev_etype is not None:
                edge_indices[rev_etype][0].append(dst_idx)
                edge_indices[rev_etype][1].append(src_idx)

    for pyg_etype, (src_list, dst_list) in edge_indices.items():
        if src_list:
            data[pyg_etype].edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
        else:
            # Empty edge type — still create a valid empty tensor.
            data[pyg_etype].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

    click.echo(
        f"  Shows: {num_shows}, Songs: {num_songs}, "
        f"Performances: {num_perfs}"
    )
    for pyg_etype, (src_list, _) in edge_indices.items():
        if src_list:
            click.echo(f"  {pyg_etype[1]}: {len(src_list)} edges")

    return data, node_id_maps


class DeadRecsGNN(nn.Module):
    """Heterogeneous GraphSAGE model for link prediction.

    Uses HeteroConv with SAGEConv per edge type. Two message-passing
    layers with ReLU activation and dropout between them.
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # All edge types including reverse edges for bidirectional message passing.
        all_edge_types = [
            ("show", "has_performance", "performance"),
            ("performance", "of_song", "song"),
            ("song", "transitioned_to", "song"),
            ("show", "setlist_neighbor", "show"),
            ("performance", "rev_has_performance", "show"),
            ("song", "rev_of_song", "performance"),
            ("song", "rev_transitioned_to", "song"),
        ]

        # Layer 1: project each node type to hidden_dim, then message-pass.
        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in all_edge_types},
            aggr="sum",
        )

        # Layer 2: hidden_dim -> out_dim
        self.conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), out_dim) for et in all_edge_types},
            aggr="sum",
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning embeddings for each node type."""
        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)

        return x_dict
