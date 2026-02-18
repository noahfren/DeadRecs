"""GNN training loop for DeadRecs — self-supervised link prediction.

Trains a heterogeneous GraphSAGE model using link prediction on
SETLIST_NEIGHBOR and HAS_PERFORMANCE edges, with early stopping
on validation AUC.
"""

from __future__ import annotations

import logging
from typing import Any

import click
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from deadrecs.model import DeadRecsGNN
from deadrecs.utils import EMBEDDINGS_PATH, MODEL_PATH

logger = logging.getLogger(__name__)

# Edge types used for link prediction supervision
SUPERVISION_EDGE_TYPES = [
    ("show", "setlist_neighbor", "show"),
    ("show", "has_performance", "performance"),
]


def _split_edges(
    data: HeteroData,
    val_ratio: float = 0.15,
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Split edges into train/val/test sets for link prediction.

    Only SETLIST_NEIGHBOR and HAS_PERFORMANCE edges are split for
    supervision. Other edge types are kept as message-passing edges.
    """
    # Determine which edge types exist in the data
    existing_types = []
    for etype in SUPERVISION_EDGE_TYPES:
        if etype in data.edge_types:
            existing_types.append(etype)

    if not existing_types:
        raise ValueError("No supervision edge types found in data")

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=0.0,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=existing_types,
        rev_edge_types=[None] * len(existing_types),
    )
    train_data, val_data, _ = transform(data)
    return train_data, val_data, data


def _compute_link_prediction_loss(
    model: DeadRecsGNN,
    data: HeteroData,
) -> Tensor:
    """Compute binary cross-entropy loss for link prediction.

    Uses the edge_label and edge_label_index attributes added by
    RandomLinkSplit for each supervision edge type.
    """
    z_dict = model(data.x_dict, data.edge_index_dict)

    total_loss = torch.tensor(0.0, requires_grad=True)
    num_edges = 0

    for etype in SUPERVISION_EDGE_TYPES:
        if etype not in data.edge_types:
            continue

        edge_store = data[etype]
        if not hasattr(edge_store, "edge_label_index") or not hasattr(edge_store, "edge_label"):
            continue

        edge_label_index = edge_store.edge_label_index
        edge_label = edge_store.edge_label.float()

        src_type, _, dst_type = etype
        src_z = z_dict.get(src_type)
        dst_z = z_dict.get(dst_type)

        if src_z is None or dst_z is None:
            continue

        src_emb = src_z[edge_label_index[0]]
        dst_emb = dst_z[edge_label_index[1]]

        # Dot product score
        scores = (src_emb * dst_emb).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(scores, edge_label)

        total_loss = total_loss + loss * edge_label.numel()
        num_edges += edge_label.numel()

    if num_edges > 0:
        total_loss = total_loss / num_edges

    return total_loss


@torch.no_grad()
def _compute_auc(
    model: DeadRecsGNN,
    data: HeteroData,
) -> float:
    """Compute AUC for link prediction on validation data."""
    model.eval()
    z_dict = model(data.x_dict, data.edge_index_dict)

    all_scores = []
    all_labels = []

    for etype in SUPERVISION_EDGE_TYPES:
        if etype not in data.edge_types:
            continue

        edge_store = data[etype]
        if not hasattr(edge_store, "edge_label_index") or not hasattr(edge_store, "edge_label"):
            continue

        edge_label_index = edge_store.edge_label_index
        edge_label = edge_store.edge_label.float()

        src_type, _, dst_type = etype
        src_z = z_dict.get(src_type)
        dst_z = z_dict.get(dst_type)

        if src_z is None or dst_z is None:
            continue

        src_emb = src_z[edge_label_index[0]]
        dst_emb = dst_z[edge_label_index[1]]
        scores = (src_emb * dst_emb).sum(dim=-1)

        all_scores.append(scores)
        all_labels.append(edge_label)

    if not all_scores:
        return 0.5

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # Simple AUC computation without sklearn dependency
    return _manual_auc(scores, labels)


def _manual_auc(scores: Tensor, labels: Tensor) -> float:
    """Compute AUC using the Wilcoxon-Mann-Whitney statistic."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Count pairs where positive score > negative score
    correct = 0
    ties = 0
    total = len(pos_scores) * len(neg_scores)

    for ps in pos_scores:
        correct += (neg_scores < ps).sum().item()
        ties += (neg_scores == ps).sum().item()

    return (correct + 0.5 * ties) / total


def train_gnn(
    data: HeteroData,
    epochs: int = 100,
    lr: float = 0.01,
    hidden_dim: int = 128,
    out_dim: int = 128,
    patience: int = 10,
    val_ratio: float = 0.15,
) -> tuple[DeadRecsGNN, dict[str, Tensor]]:
    """Train the GNN model and return it with node embeddings.

    Args:
        data: PyG HeteroData with node features and edge indices.
        epochs: Maximum training epochs.
        lr: Learning rate.
        hidden_dim: Hidden layer dimension.
        out_dim: Output embedding dimension.
        patience: Early stopping patience (epochs without improvement).
        val_ratio: Fraction of edges to use for validation.

    Returns:
        model: Trained DeadRecsGNN model.
        embeddings: Dict mapping node type -> embedding tensor.
    """
    click.echo("\nPreparing edge splits for link prediction...")
    train_data, val_data, _ = _split_edges(data, val_ratio=val_ratio)

    metadata = data.metadata()
    model = DeadRecsGNN(
        metadata=metadata,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    click.echo(f"Training GNN for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        loss = _compute_link_prediction_loss(model, train_data)
        loss.backward()
        optimizer.step()

        # Validate
        val_auc = _compute_auc(model, val_data)

        if epoch % 10 == 1 or epoch == epochs:
            click.echo(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"loss={loss.item():.4f}  "
                f"val_AUC={val_auc:.4f}"
            )

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                click.echo(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best val AUC={best_val_auc:.4f} at epoch {best_epoch})"
                )
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    click.echo(f"\nBest validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)

    return model, embeddings


def save_model(model: DeadRecsGNN) -> None:
    """Save the trained model weights to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    click.echo(f"Model saved to {MODEL_PATH}")


def save_embeddings(
    embeddings: dict[str, Tensor],
    node_id_maps: dict[str, dict[str, int]],
) -> None:
    """Save node embeddings to disk with ID mappings.

    Saves a dict mapping node IDs (e.g. 'show:1977-05-08') to
    embedding vectors, for each node type.
    """
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Invert node_id_maps: pyg_index -> nx_node_id
    result: dict[str, Tensor] = {}
    for ntype, id_map in node_id_maps.items():
        if ntype not in embeddings:
            continue
        emb_tensor = embeddings[ntype]
        for nx_id, pyg_idx in id_map.items():
            if pyg_idx < emb_tensor.shape[0]:
                result[nx_id] = emb_tensor[pyg_idx]

    torch.save(result, EMBEDDINGS_PATH)
    click.echo(f"Embeddings saved to {EMBEDDINGS_PATH} ({len(result)} nodes)")
