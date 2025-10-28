from __future__ import annotations

from torch import Tensor, nn


class MFRecommender(nn.Module):
    """Matrix factorization recommender built on top of learnable embeddings."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 100) -> None:
        """
        Args:
            num_users: Number of distinct users (size of the user embedding table).
            num_items: Number of distinct items (size of the item embedding table).
            embedding_dim: Latent dimension used for both embeddings.
        """
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Encourage symmetry around zero for initial predictions.
        self.user_embedding.weight.data.uniform_(0, 0.05)
        self.item_embedding.weight.data.uniform_(0, 0.05)

    def forward(self, user_indices: Tensor, item_indices: Tensor) -> Tensor:
        """Predict ratings for the provided user-item pairs."""
        user_vectors = self.user_embedding(user_indices)
        item_vectors = self.item_embedding(item_indices)
        return (user_vectors * item_vectors).sum(dim=1)
