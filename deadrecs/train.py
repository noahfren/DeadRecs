"""GNN training loop with self-supervised link prediction.

Trains the DeadRecsGNN model using link prediction on the heterogeneous
graph edges. Uses edge-split validation with early stopping on AUC.
"""

from __future__ import annotations

import logging

import click
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData

from deadrecs.model import DeadRecsGNN, convert_to_hetero_data
from deadrecs.utils import EMBEDDINGS_PATH, MODEL_PATH

logger = logging.getLogger(__name__)

# Edge types used for link prediction training.
SUPERVISED_EDGE_TYPES = [
    ("show", "has_performance", "performance"),
    ("performance", "of_song", "song"),
    ("show", "setlist_neighbor", "show"),
]


def _split_edges(
    data: HeteroData,
    val_ratio: float = 0.15,
) -> tuple[HeteroData, dict[tuple, torch.Tensor], dict[tuple, torch.Tensor]]:
    """Split edges into train and validation sets.

    Returns:
        train_data: HeteroData with only training edges.
        val_pos: Dict mapping edge type -> positive validation edge_index.
        val_neg: Dict mapping edge type -> negative validation edge_index.
    """
    train_data = data.clone()
    val_pos: dict[tuple, torch.Tensor] = {}
    val_neg: dict[tuple, torch.Tensor] = {}

    for etype in SUPERVISED_EDGE_TYPES:
        if etype not in data.edge_types:
            continue
        edge_index = data[etype].edge_index
        num_edges = edge_index.size(1)
        if num_edges < 2:
            continue

        # Shuffle and split
        perm = torch.randperm(num_edges)
        num_val = max(1, int(num_edges * val_ratio))
        val_idx = perm[:num_val]
        train_idx = perm[num_val:]

        val_pos[etype] = edge_index[:, val_idx]
        train_data[etype].edge_index = edge_index[:, train_idx]

        # Negative sampling: random edges of the same type
        src_type, _, dst_type = etype
        num_src = data[src_type].num_nodes
        num_dst = data[dst_type].num_nodes
        neg_src = torch.randint(0, num_src, (num_val,))
        neg_dst = torch.randint(0, num_dst, (num_val,))
        val_neg[etype] = torch.stack([neg_src, neg_dst])

    return train_data, val_pos, val_neg


def _negative_sample(
    edge_index: torch.Tensor,
    num_src: int,
    num_dst: int,
) -> torch.Tensor:
    """Generate negative edge samples of the same size as edge_index."""
    num_edges = edge_index.size(1)
    neg_src = torch.randint(0, num_src, (num_edges,))
    neg_dst = torch.randint(0, num_dst, (num_edges,))
    return torch.stack([neg_src, neg_dst])


def _compute_link_score(
    embeddings: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    src_type: str,
    dst_type: str,
) -> torch.Tensor:
    """Compute cosine-similarity link scores for edges."""
    src_emb = F.normalize(embeddings[src_type][edge_index[0]], dim=1)
    dst_emb = F.normalize(embeddings[dst_type][edge_index[1]], dim=1)
    return (src_emb * dst_emb).sum(dim=1)


def _compute_loss(
    embeddings: dict[str, torch.Tensor],
    train_data: HeteroData,
) -> torch.Tensor:
    """Compute binary cross-entropy loss over positive and negative edges."""
    total_loss = torch.tensor(0.0)
    num_edge_types = 0

    for etype in SUPERVISED_EDGE_TYPES:
        if etype not in train_data.edge_types:
            continue
        edge_index = train_data[etype].edge_index
        if edge_index.size(1) == 0:
            continue

        src_type, _, dst_type = etype

        # Positive scores
        pos_scores = _compute_link_score(embeddings, edge_index, src_type, dst_type)

        # Negative samples
        neg_edge_index = _negative_sample(
            edge_index,
            train_data[src_type].num_nodes,
            train_data[dst_type].num_nodes,
        )
        neg_scores = _compute_link_score(embeddings, neg_edge_index, src_type, dst_type)

        # BCEWithLogits
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        total_loss = total_loss + pos_loss + neg_loss
        num_edge_types += 1

    if num_edge_types > 0:
        total_loss = total_loss / num_edge_types

    return total_loss


def _compute_val_auc(
    embeddings: dict[str, torch.Tensor],
    val_pos: dict[tuple, torch.Tensor],
    val_neg: dict[tuple, torch.Tensor],
) -> float:
    """Compute validation AUC across all supervised edge types."""
    all_scores = []
    all_labels = []

    for etype in SUPERVISED_EDGE_TYPES:
        if etype not in val_pos:
            continue
        src_type, _, dst_type = etype

        pos_ei = val_pos[etype]
        neg_ei = val_neg[etype]

        if pos_ei.size(1) == 0:
            continue

        pos_scores = _compute_link_score(embeddings, pos_ei, src_type, dst_type)
        neg_scores = _compute_link_score(embeddings, neg_ei, src_type, dst_type)

        scores = torch.cat([pos_scores, neg_scores]).detach().cpu().numpy()
        labels = torch.cat([
            torch.ones(pos_scores.size(0)),
            torch.zeros(neg_scores.size(0)),
        ]).numpy()

        all_scores.extend(scores)
        all_labels.extend(labels)

    if len(set(all_labels)) < 2:
        return 0.5

    return roc_auc_score(all_labels, all_scores)


def train_model(
    G: nx.DiGraph,
    epochs: int = 100,
    lr: float = 0.0005,
    patience: int = 10,
    hidden_dim: int = 128,
    out_dim: int = 128,
) -> tuple[DeadRecsGNN, dict[str, torch.Tensor]]:
    """Train the GNN model on the graph.

    Returns:
        model: Trained DeadRecsGNN model.
        all_embeddings: Dict mapping original node IDs to embedding vectors.
    """
    data, node_id_maps = convert_to_hetero_data(G)

    # Split edges
    train_data, val_pos, val_neg = _split_edges(data)

    # Initialize model
    model = DeadRecsGNN(hidden_dim=hidden_dim, out_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    click.echo(f"\nTraining GNN for up to {epochs} epochs (patience={patience})...")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()

        embeddings = model(train_data.x_dict, train_data.edge_index_dict)
        loss = _compute_loss(embeddings, train_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_embeddings = model(data.x_dict, data.edge_index_dict)
            auc = _compute_val_auc(val_embeddings, val_pos, val_neg)

        if epoch % 10 == 0 or epoch == 1:
            click.echo(
                f"  Epoch {epoch:3d}/{epochs} — "
                f"loss: {loss.item():.4f}, val AUC: {auc:.4f}"
            )

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            click.echo(
                f"  Early stopping at epoch {epoch} "
                f"(best AUC: {best_auc:.4f} at epoch {best_epoch})"
            )
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    click.echo(f"Best validation AUC: {best_auc:.4f} (epoch {best_epoch})")

    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x_dict, data.edge_index_dict)

    # Map back to original node IDs
    all_embeddings: dict[str, torch.Tensor] = {}
    for ntype, id_map in node_id_maps.items():
        if ntype not in final_embeddings:
            continue
        emb_matrix = final_embeddings[ntype]
        for nx_id, idx in id_map.items():
            all_embeddings[nx_id] = emb_matrix[idx]

    # Save model and embeddings
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    click.echo(f"Model saved to {MODEL_PATH}")

    torch.save(all_embeddings, EMBEDDINGS_PATH)
    click.echo(f"Embeddings saved to {EMBEDDINGS_PATH} ({len(all_embeddings)} nodes)")

    return model, all_embeddings
