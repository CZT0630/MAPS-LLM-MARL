# environment/channel_model.py
"""
无线信道模型 — Phase 2 实现

Shannon 无线速率模型：
  R = B * log2(1 + p * h / (N0 * B))

第一版采用正交接入（干扰项为 0），可在论文中明确说明。

NOTE: 本文件由人工完成 Phase 2 初稿，已由 Codex 于 2026-06-11 审查并修复。
"""

from __future__ import annotations

import math

class WirelessChannel:
    """UE 到 ES 的无线信道模型。

    Parameters
    ----------
    bandwidth_hz : float
        分配带宽 (Hz)，典型值 1e6 ~ 20e6。
    transmit_power_w : float
        UE 发射功率 (W)。
    noise_psd_w_hz : float
        噪声功率谱密度 N0 (W/Hz)，典型值 1e-17 ~ 1e-20。
    path_loss_exponent : float
        路径损耗指数，典型值 2.0 ~ 4.0。
    ref_distance_m : float
        参考距离 d0 (m)，路径损耗在此距离处为 0 dB。
    ref_distance_gain_db : float
        参考距离处的路径损耗 (dB)，默认 0。
    """

    def __init__(
        self,
        bandwidth_hz: float = 1e6,
        transmit_power_w: float = 0.5,
        noise_psd_w_hz: float = 1e-17,
        path_loss_exponent: float = 3.0,
        ref_distance_m: float = 1.0,
        ref_distance_gain_db: float = 0.0,
    ):
        self.bandwidth_hz = float(bandwidth_hz)
        self.transmit_power_w = float(transmit_power_w)
        self.noise_psd_w_hz = float(noise_psd_w_hz)
        self.path_loss_exponent = float(path_loss_exponent)
        self.ref_distance_m = float(ref_distance_m)
        self.ref_distance_gain_linear = 10.0 ** (float(ref_distance_gain_db) / 10.0)
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        if self.transmit_power_w < 0:
            raise ValueError("transmit_power_w must be non-negative")
        if self.noise_psd_w_hz <= 0:
            raise ValueError("noise_psd_w_hz must be positive")
        if self.path_loss_exponent <= 0:
            raise ValueError("path_loss_exponent must be positive")
        if self.ref_distance_m <= 0:
            raise ValueError("ref_distance_m must be positive")

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------

    def path_loss(self, distance_m: float) -> float:
        """返回线性域路径损耗 (<=1 的衰减因子)。

        PL(d) = (d0/d)^n * G0
        """
        distance_m = max(float(distance_m), self.ref_distance_m)
        ratio = self.ref_distance_m / distance_m
        return (ratio ** self.path_loss_exponent) * self.ref_distance_gain_linear

    def channel_gain(self, distance_m: float, fading: float = 1.0) -> float:
        """信道增益 h = PL(d) * |fading|^2。

        Parameters
        ----------
        distance_m : float
            UE 与 ES 之间的距离 (m)。
        fading : float
            小尺度衰落幅度系数 (线性域)，默认 1.0 (无衰落)。
        """
        return self.path_loss(distance_m) * (fading ** 2)

    def noise_power(self) -> float:
        """总噪声功率 = N0 * B (W)。"""
        return self.noise_psd_w_hz * self.bandwidth_hz

    def achievable_rate(
        self,
        distance_m: float,
        fading: float = 1.0,
        interference_w: float = 0.0,
    ) -> float:
        """Shannon 速率 (bits/s)。

        R = B * log2(1 + p * h / (N0 * B + I))

        Parameters
        ----------
        distance_m : float
            UE 与 ES 之间的距离 (m)。
        fading : float
            小尺度衰落幅度系数。
        interference_w : float
            干扰功率 (W)，正交接入时为 0。
        """
        if interference_w < 0:
            raise ValueError("interference_w must be non-negative")
        h = self.channel_gain(distance_m, fading)
        noise = self.noise_power()
        sinr = (self.transmit_power_w * h) / (noise + interference_w)
        return self.bandwidth_hz * math.log2(1.0 + sinr)

    def transmission_time(
        self,
        data_size_bits: float,
        distance_m: float,
        fading: float = 1.0,
    ) -> float:
        """传输时延 = 数据量 / 速率 (s)。

        Parameters
        ----------
        data_size_bits : float
            传输数据量 (bits)。
        distance_m : float
            UE 与 ES 之间的距离 (m)。
        fading : float
            小尺度衰落幅度系数。
        """
        rate = self.achievable_rate(distance_m, fading)
        if rate <= 0:
            return float("inf")
        return data_size_bits / rate

    def transmission_energy(
        self,
        data_size_bits: float,
        distance_m: float,
        fading: float = 1.0,
    ) -> float:
        """传输能耗 = 功率 × 时延 (J)。"""
        t = self.transmission_time(data_size_bits, distance_m, fading)
        return self.transmit_power_w * t

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "bandwidth_hz": self.bandwidth_hz,
            "transmit_power_w": self.transmit_power_w,
            "noise_psd_w_hz": self.noise_psd_w_hz,
            "path_loss_exponent": self.path_loss_exponent,
            "ref_distance_m": self.ref_distance_m,
            "ref_distance_gain_db": 10.0 * math.log10(self.ref_distance_gain_linear)
            if self.ref_distance_gain_linear > 0
            else float("-inf"),
        }


def build_channel_from_config(config: dict) -> WirelessChannel:
    """从配置字典构建 WirelessChannel 实例。

    配置键位于 ``config['channel']`` 下。
    """
    cfg = config.get("channel", {})
    return WirelessChannel(
        bandwidth_hz=float(cfg.get("bandwidth_hz", 1e6)),
        transmit_power_w=float(cfg.get("transmit_power_w", 0.5)),
        noise_psd_w_hz=float(cfg.get("noise_psd_w_hz", 1e-17)),
        path_loss_exponent=float(cfg.get("path_loss_exponent", 3.0)),
        ref_distance_m=float(cfg.get("ref_distance_m", 1.0)),
        ref_distance_gain_db=float(cfg.get("ref_distance_gain_db", 0.0)),
    )
