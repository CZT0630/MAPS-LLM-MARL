"""Unified hybrid action codec for MAPS EI.

Each UE action is defined as:
    a_u = (phi_u, p_u)
    phi_u = softmax(partition_logits)       # 3 dims: local, edge, cloud
    p_u   = softmax(edge_logits)            # E dims: edge server probabilities

The codec provides:
  - logit-to-probability conversion (for actor output → replay/critic/env)
  - probability-to-env-action conversion (for env boundary: argmax edge)
  - env-action-to-policy conversion (for expert cache → replay storage)
  - hybrid distillation loss (partition MSE + edge CE)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


class HybridActionCodec:
    """Stateless codec for hybrid partition + edge-select actions.

    Policy action layout: [phi_local, phi_edge, phi_cloud, p_edge_0, ..., p_edge_{E-1}]
      - First 3 dims: partition probability ratios (sum to 1)
      - Last E dims: edge server probability distribution (sum to 1)
    """

    def __init__(self, num_edges: int):
        try:
            parsed_num_edges = int(num_edges)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"num_edges must be an integer, got {num_edges!r}") from exc
        if (
            isinstance(num_edges, bool)
            or parsed_num_edges != num_edges
            or parsed_num_edges < 1
        ):
            raise ValueError(f"num_edges must be >= 1, got {num_edges}")
        self.num_edges = parsed_num_edges
        self.partition_dim = 3
        self.edge_dim = self.num_edges
        self.action_dim = self.partition_dim + self.edge_dim

    # ------------------------------------------------------------------
    # Actor output → policy action (probabilities)
    # ------------------------------------------------------------------

    def logits_to_probs(
        self,
        partition_logits: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Convert raw logits to concatenated probability vector.

        Parameters
        ----------
        partition_logits : Tensor[..., 3]
        edge_logits : Tensor[..., E]

        Returns
        -------
        Tensor[..., 3+E]  probabilities
        """
        if partition_logits.shape[-1] != self.partition_dim:
            raise ValueError(
                "partition_logits must end with 3 values, "
                f"got shape {tuple(partition_logits.shape)}"
            )
        if edge_logits.shape[-1] != self.edge_dim:
            raise ValueError(
                f"edge_logits must end with {self.edge_dim} values, "
                f"got shape {tuple(edge_logits.shape)}"
            )
        if partition_logits.shape[:-1] != edge_logits.shape[:-1]:
            raise ValueError(
                "partition_logits and edge_logits batch shapes differ: "
                f"{tuple(partition_logits.shape)} vs {tuple(edge_logits.shape)}"
            )
        if not torch.isfinite(partition_logits).all():
            raise ValueError("partition_logits contain non-finite values")
        if not torch.isfinite(edge_logits).all():
            raise ValueError("edge_logits contain non-finite values")
        phi = F.softmax(partition_logits, dim=-1)
        p = F.softmax(edge_logits, dim=-1)
        return torch.cat([phi, p], dim=-1)

    def partition_concentration_from_logits(
        self,
        partition_logits: torch.Tensor,
        total_concentration: float,
    ) -> torch.Tensor:
        """Build strictly positive Dirichlet parameters from partition logits."""
        if partition_logits.shape[-1] != self.partition_dim:
            raise ValueError(
                "partition_logits must end with 3 values, "
                f"got shape {tuple(partition_logits.shape)}"
            )
        if not torch.is_floating_point(partition_logits):
            raise ValueError("partition_logits must be floating point")
        if not torch.isfinite(partition_logits).all():
            raise ValueError("partition_logits contain non-finite values")
        total = float(total_concentration)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("total_concentration must be finite and positive")

        probabilities = F.softmax(partition_logits, dim=-1)
        probability_floor = torch.finfo(probabilities.dtype).eps
        probabilities = probabilities.clamp_min(probability_floor)
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        )
        return probabilities * total

    # ------------------------------------------------------------------
    # Policy action → environment action
    # ------------------------------------------------------------------

    def policy_to_env_action(self, policy_action) -> np.ndarray:
        """Convert policy probabilities to env action [alpha1, alpha2, alpha3, edge_id].

        Parameters
        ----------
        policy_action : array-like, shape (3+E,)
            First 3 are partition ratios, last E are edge probabilities.

        Returns
        -------
        np.ndarray, shape (4,)
            [alpha1, alpha2, alpha3, edge_id_float]
        """
        a = np.asarray(policy_action, dtype=np.float32)
        self._validate_policy_actions(a, batched=False)
        partition = a[:3]
        edge_probs = a[3:]
        edge_id = int(np.argmax(edge_probs))
        return np.array(
            [partition[0], partition[1], partition[2], float(edge_id)],
            dtype=np.float32,
        )

    def batch_policy_to_env_actions(self, policy_actions) -> np.ndarray:
        """Convert batch of policy actions to env actions.

        Parameters
        ----------
        policy_actions : array-like, shape (N, 3+E)

        Returns
        -------
        np.ndarray, shape (N, 4)
        """
        pa = np.asarray(policy_actions, dtype=np.float32)
        self._validate_policy_actions(pa, batched=True)
        n = pa.shape[0]
        env_actions = np.empty((n, 4), dtype=np.float32)
        env_actions[:, :3] = pa[:, :3]
        env_actions[:, 3] = np.argmax(pa[:, 3:], axis=1).astype(np.float32)
        return env_actions

    # ------------------------------------------------------------------
    # Environment action → policy action (for expert cache)
    # ------------------------------------------------------------------

    def env_to_policy_action(self, env_action) -> np.ndarray:
        """Convert env action [alpha1, alpha2, alpha3, edge_id] to policy format.

        The edge_id is converted to a one-hot probability vector.

        Parameters
        ----------
        env_action : array-like, shape (4,)

        Returns
        -------
        np.ndarray, shape (3+E,)
        """
        a = np.asarray(env_action, dtype=np.float32)
        if a.shape != (4,):
            raise ValueError(f"env action must have shape (4,), got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("env action contains non-finite values")
        partition = a[:3]
        self._validate_probability_group(partition, "partition")
        edge_value = float(a[3])
        edge_id = int(round(edge_value))
        if not np.isclose(edge_value, edge_id, atol=1e-6):
            raise ValueError(f"edge_id must be an integer, got {edge_value}")
        if not 0 <= edge_id < self.num_edges:
            raise ValueError(
                f"edge_id must be in [0, {self.num_edges - 1}], got {edge_id}"
            )
        edge_onehot = np.zeros(self.num_edges, dtype=np.float32)
        edge_onehot[edge_id] = 1.0
        return np.concatenate([partition, edge_onehot])

    def _validate_policy_actions(
        self, policy_actions: np.ndarray, *, batched: bool
    ) -> None:
        expected_shape = (
            ("N", self.action_dim) if batched else (self.action_dim,)
        )
        if batched:
            if policy_actions.ndim != 2 or policy_actions.shape[1] != self.action_dim:
                raise ValueError(
                    "policy actions must have shape "
                    f"(N, {self.action_dim}), got {policy_actions.shape}"
                )
            if policy_actions.shape[0] == 0:
                raise ValueError("policy action batch must not be empty")
            rows = policy_actions
        else:
            if policy_actions.shape != expected_shape:
                raise ValueError(
                    f"policy action must have shape {expected_shape}, "
                    f"got {policy_actions.shape}"
                )
            rows = policy_actions[None, :]

        if not np.isfinite(rows).all():
            raise ValueError("policy actions contain non-finite values")
        for row_index, row in enumerate(rows):
            self._validate_probability_group(
                row[: self.partition_dim], f"partition row {row_index}"
            )
            self._validate_probability_group(
                row[self.partition_dim :], f"edge row {row_index}"
            )

    def validate_policy_tensor(
        self,
        policy_actions: torch.Tensor,
        *,
        name: str = "policy_actions",
    ) -> None:
        """Validate a batched differentiable policy tensor without detaching it."""
        if policy_actions.ndim != 2 or policy_actions.shape[1] != self.action_dim:
            raise ValueError(
                f"{name} must have shape (B, {self.action_dim}), "
                f"got {tuple(policy_actions.shape)}"
            )
        if policy_actions.shape[0] == 0:
            raise ValueError(f"{name} batch must not be empty")
        if not torch.is_floating_point(policy_actions):
            raise ValueError(f"{name} must be floating point")
        if not torch.isfinite(policy_actions).all():
            raise ValueError(f"{name} contains non-finite values")
        if torch.any(policy_actions < -1e-6):
            raise ValueError(f"{name} contains negative probabilities")

        partition_sums = policy_actions[:, :3].sum(dim=-1)
        edge_sums = policy_actions[:, 3:].sum(dim=-1)
        expected = torch.ones_like(partition_sums)
        if not torch.allclose(partition_sums, expected, atol=1e-5, rtol=0):
            raise ValueError(f"{name} partition probabilities must sum to 1")
        if not torch.allclose(edge_sums, expected, atol=1e-5, rtol=0):
            raise ValueError(f"{name} edge probabilities must sum to 1")

    @staticmethod
    def _validate_probability_group(values: np.ndarray, name: str) -> None:
        if np.any(values < -1e-6):
            raise ValueError(f"{name} contains negative probabilities")
        if not np.isclose(float(values.sum()), 1.0, atol=1e-5):
            raise ValueError(f"{name} probabilities must sum to 1")

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    def distillation_loss(
        self,
        actor_probs: torch.Tensor,
        expert_partition: torch.Tensor,
        expert_edge_id: torch.Tensor,
        expert_mask: torch.Tensor,
        eta_edge: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mixed distillation loss: partition squared L2 + edge CE.

        Parameters
        ----------
        actor_probs : Tensor[B, 3+E]
            Actor output probabilities (from softmax).
        expert_partition : Tensor[B, 3]
            Expert partition ratios (already normalized).
        expert_edge_id : Tensor[B]
            Expert edge selection as integer index (long tensor).
        expert_mask : Tensor[B]
            1.0 for valid experts, 0.0 for missing/invalid.
        eta_edge : float
            Weight for edge CE relative to partition MSE.

        Returns
        -------
        total_distill : Tensor scalar
        L_part : Tensor scalar   (partition squared L2)
        L_edge : Tensor scalar   (edge CE)
        """
        self.validate_policy_tensor(actor_probs, name="actor_probs")
        batch_size = actor_probs.shape[0]
        expected_partition_shape = (batch_size, self.partition_dim)
        if tuple(expert_partition.shape) != expected_partition_shape:
            raise ValueError(
                "expert_partition must have shape "
                f"{expected_partition_shape}, got {tuple(expert_partition.shape)}"
            )
        if tuple(expert_edge_id.shape) != (batch_size,):
            raise ValueError(
                f"expert_edge_id must have shape ({batch_size},), "
                f"got {tuple(expert_edge_id.shape)}"
            )
        if tuple(expert_mask.shape) != (batch_size,):
            raise ValueError(
                f"expert_mask must have shape ({batch_size},), "
                f"got {tuple(expert_mask.shape)}"
            )
        tensors = (expert_partition, expert_edge_id, expert_mask)
        if any(tensor.device != actor_probs.device for tensor in tensors):
            raise ValueError("all distillation tensors must use the same device")
        if not torch.is_floating_point(expert_partition):
            raise ValueError("expert_partition must be floating point")
        if not torch.isfinite(expert_partition).all():
            raise ValueError("expert_partition contains non-finite values")
        if torch.any(expert_partition < -1e-6):
            raise ValueError("expert_partition contains negative probabilities")
        partition_sums = expert_partition.sum(dim=-1)
        if not torch.allclose(
            partition_sums,
            torch.ones_like(partition_sums),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("expert_partition probabilities must sum to 1")

        edge_values = expert_edge_id.to(dtype=torch.float32)
        if not torch.isfinite(edge_values).all():
            raise ValueError("expert_edge_id contains non-finite values")
        if not torch.equal(edge_values, edge_values.round()):
            raise ValueError("expert_edge_id values must be integers")
        if torch.any(edge_values < 0) or torch.any(edge_values >= self.num_edges):
            raise ValueError(
                f"expert_edge_id values must be in [0, {self.num_edges - 1}]"
            )
        edge_indices = edge_values.long()

        mask = expert_mask.float()
        if not torch.isfinite(mask).all():
            raise ValueError("expert_mask contains non-finite values")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("expert_mask values must be binary 0 or 1")
        eta = float(eta_edge)
        if not math.isfinite(eta) or eta < 0:
            raise ValueError("eta_edge must be finite and non-negative")

        n_valid = mask.sum().clamp(min=1.0)

        # Partition squared L2 distance
        pred_partition = actor_probs[:, :3]
        per_sample_part = F.mse_loss(
            pred_partition, expert_partition, reduction="none"
        ).sum(dim=-1)
        L_part = (per_sample_part * mask).sum() / n_valid

        # Edge CE
        pred_edge_log_probs = torch.log(actor_probs[:, 3:].clamp(min=1e-8))
        per_sample_edge = F.nll_loss(
            pred_edge_log_probs, edge_indices, reduction="none"
        )
        L_edge = (per_sample_edge * mask).sum() / n_valid

        total = L_part + eta * L_edge
        return total, L_part, L_edge
