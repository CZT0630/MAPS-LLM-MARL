# environment/backhaul_model.py
"""
ES→CS 有线回传链路模型 — Phase 2 实现

独立于 UE 无线链路：
  - 传输时延：data_size_bits / rate_bps + propagation_latency_s
  - 网络能耗：energy_per_bit * data_size_bits

NOTE: 本文件由人工完成 Phase 2 初稿，已由 Codex 于 2026-06-11 审查并修复。
"""

from __future__ import annotations


class BackhaulLink:
    """有线回传链路 (ES→CS)。

    Parameters
    ----------
    rate_bps : float
        链路速率 (bits/s)。
    energy_per_bit : float
        每 bit 传输能耗 (J/bit)。
    propagation_latency_s : float
        固定传播时延 (s)，典型值 0.001 ~ 0.01。
    """

    def __init__(
        self,
        rate_bps: float = 1e9,
        energy_per_bit: float = 1e-9,
        propagation_latency_s: float = 0.002,
    ):
        self.rate_bps = float(rate_bps)
        self.energy_per_bit = float(energy_per_bit)
        self.propagation_latency_s = float(propagation_latency_s)
        if self.rate_bps <= 0:
            raise ValueError("rate_bps must be positive")
        if self.energy_per_bit < 0:
            raise ValueError("energy_per_bit must be non-negative")
        if self.propagation_latency_s < 0:
            raise ValueError("propagation_latency_s must be non-negative")

    def transmission_time(self, data_size_bits: float) -> float:
        """传输时延 = propagation + data / rate (s)。"""
        if data_size_bits <= 0:
            return 0.0
        return self.propagation_latency_s + data_size_bits / self.rate_bps

    def transmission_energy(self, data_size_bits: float) -> float:
        """传输能耗 = energy_per_bit * data_size_bits (J)。"""
        if data_size_bits <= 0:
            return 0.0
        return self.energy_per_bit * data_size_bits

    def to_dict(self) -> dict:
        return {
            "rate_bps": self.rate_bps,
            "energy_per_bit": self.energy_per_bit,
            "propagation_latency_s": self.propagation_latency_s,
        }


def build_backhaul_from_config(config: dict) -> BackhaulLink:
    """从配置字典构建 BackhaulLink 实例。

    配置键位于 ``config['backhaul']`` 下。
    """
    cfg = config.get("backhaul", {})
    return BackhaulLink(
        rate_bps=float(cfg.get("rate_bps", 1e9)),
        energy_per_bit=float(cfg.get("energy_per_bit", 1e-9)),
        propagation_latency_s=float(cfg.get("propagation_latency_s", 0.002)),
    )
