from __future__ import annotations

from typing import Iterable, Sequence

import torch
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
        # Predicted rating is the dot product of user and item embeddings.
        return (user_vectors * item_vectors).sum(1)

    def score_user_items(self, user_index: int, item_indices: Sequence[int]) -> Tensor:
        """
        Predict ratings for a single user against multiple items.

        Args:
            user_index: Contiguous index of the target user.
            item_indices: Iterable of contiguous item indices to score.

        Returns:
            Tensor containing the predicted ratings in the same order as item_indices.
        """
        if not item_indices:
            return torch.empty(
                0, dtype=self.user_embedding.weight.dtype, device=self.user_embedding.weight.device
            )

        device = self.user_embedding.weight.device
        items = torch.as_tensor(item_indices, dtype=torch.long, device=device)
        users = torch.full_like(items, fill_value=user_index)
        return self.forward(users, items)

    def recommend_for_user(
        self,
        user_index: int,
        *,
        top_k: int = 10,
        exclude_items: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Recommend the top-k items for a user based on learned embeddings.

        Args:
            user_index: Contiguous index of the target user.
            top_k: Number of items to return.
            exclude_items: Optional iterable of item indices to ignore.

        Returns:
            List of (item_index, score) tuples ordered by descending score.
        """
        if top_k <= 0:
            return []

        device = self.user_embedding.weight.device
        num_items = self.item_embedding.num_embeddings
        with torch.no_grad():
            user_vector = self.user_embedding(torch.tensor(user_index, device=device))
            scores = torch.mv(self.item_embedding.weight, user_vector)

            if exclude_items:
                exclusion = torch.tensor(
                    sorted(set(exclude_items)),
                    dtype=torch.long,
                    device=device,
                )
                if exclusion.numel():
                    if torch.any((exclusion < 0) | (exclusion >= num_items)):
                        raise ValueError("exclude_items contains indices outside valid range.")
                    scores = scores.clone()
                    scores[exclusion] = float("-inf")

            k = min(top_k, num_items)
            top_scores, top_indices = torch.topk(scores, k)

        return [
            (int(idx), float(score))
            for idx, score in zip(top_indices.cpu(), top_scores.cpu())
        ]
