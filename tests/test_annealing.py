"""Tests for C2: mixed distillation annealing schedule.

Validates:
  - AnnealingSchedule three-stage piecewise-constant behaviour
  - FixedSchedule constant lambda
  - lambda=0 produces no distillation gradients in MADDPG actor
  - L_part and L_edge are recorded separately
  - maps / maps_no_annealing algorithm modes integrate correctly
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from LLM4RL.algos.common.annealing import AnnealingSchedule, FixedSchedule
from LLM4RL.algos.maddpg.maddpg_agent import MADDPGAgent
from LLM4RL.experiments.runner import (
    PHASE1_ALGORITHMS,
    SUPPORTED_ALGORITHMS,
    run_baseline,
)
from LLM4RL.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


# ------------------------------------------------------------------
# AnnealingSchedule unit tests
# ------------------------------------------------------------------


class TestAnnealingSchedule:
    def test_stage1_high_lambda(self):
        s = AnnealingSchedule(lambda_high=0.8, lambda_low=0.15, stage1_end=0.3)
        assert s.get_lambda(0.0) == 0.8
        assert s.get_lambda(0.1) == 0.8
        assert s.get_lambda(0.29) == 0.8

    def test_stage2_low_lambda(self):
        s = AnnealingSchedule(lambda_high=0.8, lambda_low=0.15, stage1_end=0.3)
        assert s.get_lambda(0.3) == 0.15
        assert s.get_lambda(0.5) == 0.15
        assert s.get_lambda(0.69) == 0.15

    def test_stage3_zero(self):
        s = AnnealingSchedule(lambda_high=0.8, lambda_low=0.15, stage1_end=0.3)
        assert s.get_lambda(0.7) == 0.0
        assert s.get_lambda(0.9) == 0.0
        assert s.get_lambda(1.0) == 0.0

    def test_boundary_values(self):
        s = AnnealingSchedule(
            lambda_high=0.5,
            lambda_low=0.1,
            stage1_end=0.5,
            stage2_end=0.5,
        )
        assert s.get_lambda(0.49) == 0.5
        assert s.get_lambda(0.5) == 0.0

    def test_custom_thresholds(self):
        s = AnnealingSchedule(
            lambda_high=1.0,
            lambda_low=0.5,
            stage1_end=0.5,
            stage2_end=0.8,
        )
        assert s.get_lambda(0.0) == 1.0
        assert s.get_lambda(0.5) == 0.5
        assert s.get_lambda(0.8) == 0.0

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            AnnealingSchedule(stage1_end=0.7, stage2_end=0.3)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            AnnealingSchedule(lambda_high=-1.0)
        with pytest.raises(ValueError):
            AnnealingSchedule(lambda_high=0.1, lambda_low=0.2)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_rejects_non_finite_values(self, value):
        with pytest.raises(ValueError):
            AnnealingSchedule(lambda_high=value)
        with pytest.raises(ValueError):
            AnnealingSchedule(stage1_end=value)

    @pytest.mark.parametrize("progress", [-0.01, 1.01, float("nan")])
    def test_rejects_invalid_progress(self, progress):
        with pytest.raises(ValueError):
            AnnealingSchedule().get_lambda(progress)

    def test_serializes_reproducible_configuration(self):
        schedule = AnnealingSchedule(0.8, 0.15, 0.3, 0.7)
        assert schedule.to_dict() == {
            "type": "annealing",
            "lambda_high": 0.8,
            "lambda_low": 0.15,
            "stage1_end": 0.3,
            "stage2_end": 0.7,
            "progress_unit": "environment_steps",
        }


class TestFixedSchedule:
    def test_constant_lambda(self):
        s = FixedSchedule(lambda_value=0.25)
        for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            assert s.get_lambda(p) == 0.25

    def test_zero_lambda(self):
        s = FixedSchedule(lambda_value=0.0)
        assert s.get_lambda(0.5) == 0.0

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            FixedSchedule(lambda_value=-0.1)

    def test_serializes_reproducible_configuration(self):
        assert FixedSchedule(0.25).to_dict() == {
            "type": "fixed",
            "lambda_value": 0.25,
            "progress_unit": "environment_steps",
        }


# ------------------------------------------------------------------
# MADDPG agent distillation gradient tests
# ------------------------------------------------------------------

NUM_AGENTS = 3
NUM_EDGES = 5
STATE_DIM = 5
ACTION_DIM = 3 + NUM_EDGES


def _make_batch(batch_size=4):
    """Create a minimal replay batch with valid expert actions."""
    rng = np.random.RandomState(42)
    # Construct valid probability actions: partition sums to 1, edge sums to 1
    partition = rng.dirichlet([1, 1, 1], size=batch_size * NUM_AGENTS)
    edge_onehot = np.zeros((batch_size * NUM_AGENTS, NUM_EDGES), dtype=np.float32)
    edge_idx = rng.randint(0, NUM_EDGES, size=batch_size * NUM_AGENTS)
    edge_onehot[np.arange(len(edge_idx)), edge_idx] = 1.0
    actions = np.concatenate([partition, edge_onehot], axis=1).reshape(
        batch_size, NUM_AGENTS, ACTION_DIM
    ).astype(np.float32)

    return {
        "states": rng.randn(batch_size, NUM_AGENTS, STATE_DIM).astype(np.float32),
        "actions": actions,
        "rewards": rng.randn(batch_size, NUM_AGENTS).astype(np.float32),
        "next_states": rng.randn(batch_size, NUM_AGENTS, STATE_DIM).astype(np.float32),
        "done": np.zeros(batch_size, dtype=np.float32),
        "expert_actions": actions.copy(),  # valid experts
        "expert_mask": np.ones((batch_size, NUM_AGENTS), dtype=np.float32),
    }


def _make_agents():
    return [
        MADDPGAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            num_agents=NUM_AGENTS,
            agent_idx=i,
            num_edges=NUM_EDGES,
            config={"noise_sigma": 0.0},
        )
        for i in range(NUM_AGENTS)
    ]


def test_lambda_zero_no_distill_gradient():
    """When lambda_distill=0, actor gradient must come only from policy loss."""
    agents = _make_agents()
    batch = _make_batch()
    agent = agents[0]

    # Record actor params before update
    params_before = {
        n: p.clone().detach()
        for n, p in agent.actor.named_parameters()
    }

    losses = agent.update(batch, agents, lambda_distill=0.0)

    # distill_loss and L_part/L_edge must be 0
    assert losses["distill_loss"] == 0.0
    assert losses["L_distill"] == 0.0
    assert losses["L_part"] == 0.0
    assert losses["L_edge"] == 0.0
    assert losses["lambda_distill"] == 0.0

    # Actor params must have changed (from policy loss)
    changed = False
    for n, p in agent.actor.named_parameters():
        if not torch.allclose(p, params_before[n], atol=1e-8):
            changed = True
            break
    assert changed, "actor should update from policy loss even when lambda=0"


def test_lambda_zero_update_is_independent_of_expert_actions():
    """Changing only the expert target cannot change a lambda=0 actor update."""
    torch.manual_seed(7)
    first_agents = _make_agents()
    torch.manual_seed(7)
    second_agents = _make_agents()
    first_batch = _make_batch()
    second_batch = copy.deepcopy(first_batch)
    second_batch["expert_actions"] = np.flip(
        second_batch["expert_actions"], axis=-1
    ).copy()

    first_agents[0].update(first_batch, first_agents, lambda_distill=0.0)
    second_agents[0].update(second_batch, second_agents, lambda_distill=0.0)

    for first, second in zip(
        first_agents[0].actor.parameters(),
        second_agents[0].actor.parameters(),
    ):
        assert torch.allclose(first, second, atol=1e-7, rtol=0.0)


def test_lambda_positive_records_distill_components():
    """When lambda_distill>0 and experts are valid, L_part and L_edge are recorded."""
    agents = _make_agents()
    batch = _make_batch()
    agent = agents[0]

    losses = agent.update(batch, agents, lambda_distill=0.5)

    assert losses["lambda_distill"] == 0.5
    assert losses["distill_loss"] > 0.0
    assert losses["L_distill"] == losses["distill_loss"]
    assert losses["L_part"] >= 0.0
    assert losses["L_edge"] >= 0.0


def test_expert_mask_is_applied_per_agent():
    batch = _make_batch()
    batch["expert_mask"][:, 1] = 0.0

    torch.manual_seed(11)
    valid_agents = _make_agents()
    valid_losses = valid_agents[0].update(
        batch, valid_agents, lambda_distill=0.5
    )

    torch.manual_seed(11)
    masked_agents = _make_agents()
    masked_losses = masked_agents[1].update(
        batch, masked_agents, lambda_distill=0.5
    )

    assert valid_losses["L_distill"] > 0.0
    assert masked_losses["L_distill"] == 0.0
    assert masked_losses["L_part"] == 0.0
    assert masked_losses["L_edge"] == 0.0


def test_lambda_override_takes_effect():
    """lambda_distill parameter overrides self.distill_weight."""
    agents = _make_agents()
    batch = _make_batch()
    agent = agents[0]

    # Set default to 0
    agent.distill_weight = 0.0

    # Override with non-zero
    losses = agent.update(batch, agents, lambda_distill=0.3)
    assert losses["lambda_distill"] == 0.3
    assert losses["distill_loss"] > 0.0


def test_default_lambda_uses_distill_weight():
    """When lambda_distill is None, falls back to self.distill_weight."""
    agents = _make_agents()
    batch = _make_batch()
    agent = agents[0]

    agent.distill_weight = 0.0
    losses = agent.update(batch, agents)  # no lambda_distill arg
    assert losses["lambda_distill"] == 0.0


def test_annealing_schedule_integration():
    """AnnealingSchedule produces expected lambdas across the full training."""
    schedule = AnnealingSchedule(
        lambda_high=0.8, lambda_low=0.15, stage1_end=0.3, stage2_end=0.7
    )
    agents = _make_agents()
    batch = _make_batch()
    agent = agents[0]

    # Early training (progress < 0.3): high lambda.
    losses_early = agent.update(batch, agents, lambda_distill=schedule.get_lambda(0.1))
    assert losses_early["lambda_distill"] == 0.8

    # Mid training (0.3 <= progress < 0.7): low lambda.
    losses_mid = agent.update(batch, agents, lambda_distill=schedule.get_lambda(0.5))
    assert losses_mid["lambda_distill"] == 0.15

    # Late training (progress >= 0.7): zero lambda.
    losses_late = agent.update(batch, agents, lambda_distill=schedule.get_lambda(0.9))
    assert losses_late["lambda_distill"] == 0.0
    assert losses_late["distill_loss"] == 0.0


def test_all_loss_keys_present():
    """The update() return dict includes all required keys."""
    agents = _make_agents()
    batch = _make_batch()
    losses = agents[0].update(batch, agents, lambda_distill=0.1)

    expected_keys = {
        "critic_loss", "actor_loss", "policy_loss",
        "distill_loss", "L_distill", "L_part", "L_edge", "lambda_distill",
    }
    assert expected_keys.issubset(set(losses.keys()))


def _make_runner_config():
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))
    config["maddpg"].update(
        {
            "max_episodes": 1,
            "max_steps": 2,
            "batch_size": 1,
            "train_frequency": 1,
            "distill_weight": 0.9,
        }
    )
    config["llm_maddpg"] = {
        "max_episodes": 3,
        "max_steps": 4,
        "train_frequency": 1,
        "eta_edge": 0.5,
        "expert_cache": "fixtures/legacy_expert_cache.json",
        "initial_llm_distill_weight": 0.8,
        "constant_llm_distill_weight": 0.15,
        "stage1_end_progress": 0.3,
        "stage2_end_progress": 0.7,
    }
    return config


def _load_losses(result):
    return json.loads(
        Path(result["run_dir"], "training_losses.json").read_text(
            encoding="utf-8"
        )
    )


def test_c2_modes_honor_config_and_environment_step_schedules(
    tmp_path
):
    config = _make_runner_config()

    maddpg = run_baseline("maddpg", copy.deepcopy(config), 42, tmp_path / "maddpg")
    fixed = run_baseline(
        "maps_no_annealing",
        copy.deepcopy(config),
        42,
        tmp_path / "fixed",
    )
    annealed = run_baseline("maps", copy.deepcopy(config), 42, tmp_path / "maps")

    maddpg_losses = _load_losses(maddpg)
    fixed_losses = _load_losses(fixed)
    annealed_losses = _load_losses(annealed)

    assert maddpg["episodes"] == 1
    assert {item["lambda_distill"] for item in maddpg_losses} == {0.0}

    assert fixed["episodes"] == 3
    assert fixed["extra"]["buffer_size"] == 12
    assert fixed["extra"]["annealing"] is False
    assert {item["lambda_distill"] for item in fixed_losses} == {0.15}

    assert annealed["episodes"] == 3
    assert annealed["extra"]["buffer_size"] == 12
    assert annealed["extra"]["annealing"] is True
    assert {item["lambda_distill"] for item in annealed_losses} == {
        0.0,
        0.15,
        0.8,
    }
    assert all("L_distill" in item for item in annealed_losses)
    assert annealed_losses[-1]["progress"] == 1.0


def test_c2_metadata_is_saved_to_manifest_and_checkpoint(
    tmp_path
):
    result = run_baseline("maps", _make_runner_config(), 42, tmp_path)
    run_dir = Path(result["run_dir"])
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        run_dir / "models" / "agent_0_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    distillation = manifest["distillation"]
    assert distillation["eta_edge"] == 0.5
    assert distillation["schedule"]["type"] == "annealing"
    assert distillation["schedule"]["lambda_high"] == 0.8
    assert distillation["expert"]["prompt_version"] == "legacy-smoke-v1"
    assert distillation["expert"]["cache_version"] == "legacy-smoke-v1"
    assert len(distillation["expert"]["cache_sha256"]) == 64
    assert distillation["loss_scale"]["mean_abs_policy_loss"] >= 0.0
    assert distillation["loss_scale"]["mean_L_distill"] >= 0.0
    assert (
        distillation["loss_scale"]["weighted_distill_to_policy_ratio"]
        >= 0.0
    )
    assert checkpoint["metadata"]["distillation"] == distillation
    assert checkpoint["eta_edge"] == 0.5


def test_c2_modes_do_not_expand_phase1_gate_scope():
    assert "maps" in SUPPORTED_ALGORITHMS
    assert "maps_no_annealing" in SUPPORTED_ALGORITHMS
    assert "maps" not in PHASE1_ALGORITHMS
    assert "maps_no_annealing" not in PHASE1_ALGORITHMS


def test_default_smoke_config_exercises_all_maps_schedule_stages(tmp_path):
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))
    config["maps"] = {"max_episodes": 2, "max_steps": 5}

    result = run_baseline("maps", config, 42, tmp_path)
    losses = _load_losses(result)

    assert result["updates"] > 0
    assert {item["lambda_distill"] for item in losses} == {
        0.0,
        0.15,
        0.8,
    }
