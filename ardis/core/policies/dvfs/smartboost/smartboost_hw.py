"""
SmartBoost-HW: model-based DVFS policy for ARDiS (unified, configurable version).

This single policy replaces the previous SmartBoost_dvfs_* variants. Every old
variant is now one combination of constructor options:

    core_type      "P" | "E"                one core type per experiment
    margin_mode    "none"                   trust the thermal model as is
                   "global"                 add a fixed safety margin (global_margin, degC)
                   "gmm"                    add a per-window uncertainty margin (p90 of the
                                            GMM cluster the thermal-model embedding falls in)
    tcc_enabled    bool                     emulate a thermal cap: above tcc_temp every
                                            occupied core is throttled to the minimum frequency

Algorithm (unchanged from the original variants):
    Phase 1  chip is above max_temperature -> throttle the core with the lowest boostability
    Phase 2  there is thermal headroom     -> boost the core with the highest boostability
    Phase 3  no headroom                   -> boost one core and throttle another (net gain)

Note: your model architectures live in thermal_model.py and perfromance_model.py (same
folder) and are registered under the module names the pickled models may reference,
see _register_model_classes. They must be the exact files the models were saved with.
"""

from __future__ import annotations

import logging
import math
import os
import types
from typing import Any

import joblib
import numpy as np
import torch

import ardis.config as config
from ardis.core.dvfs import DVFSAction, DVFSPolicy
from ardis.core.system_state import SystemState

from . import perfromance_model as _perf_module  # your ForecastLSTM architecture
from . import thermal_model as _thermal_module    # your LSTMForecaster architecture

log = logging.getLogger(__name__)


def _register_model_classes() -> None:
    """Make your model classes resolvable by torch.load, wherever they were defined at save time.

    The pickle stores each class as '<module>.<class>'. Depending on how the models were
    saved that module is 'thermal_model', 'perfromance_model' or '__main__'.
    """
    import sys

    sys.modules.setdefault("thermal_model", _thermal_module)
    sys.modules.setdefault("perfromance_model", _perf_module)
    main = sys.modules.get("__main__")
    if main is not None:
        for module in (_thermal_module, _perf_module):
            for name in ("LSTMForecaster", "ForecastLSTM"):
                if hasattr(module, name) and not hasattr(main, name):
                    setattr(main, name, getattr(module, name))

_EPS = 1e-9
_NUM_EPOCHS = 5  # window length expected by both models


class SmartBoostHW(DVFSPolicy):
    def __init__(
        self,
        thermal_model_path: str | os.PathLike,
        performance_model_path: str | os.PathLike,
        deployment_dir: str | os.PathLike,
        *,
        core_type: str = "E",
        margin_mode: str = "none",
        global_margin: float = 0.0,
        tcc_enabled: bool = False,
        tcc_temp: float = 41.5,
        max_temperature: float = 80,
        frequency_step_size: int = 200,
        max_steps: int = 2,
        p_core_range: list[int] | None = None,
        e_core_range: list[int] | None = None,
        base_frequency: int = 2000,
        input_scaler_file: str = "input_scaler.pkl",
        output_scaler_file: str = "output_scaler.pkl",
        gmm_file: str = "gmm_thermal_model.pkl",
        cluster_stat_file: str = "cluster_stat_thermal.pkl",
    ) -> None:
        if core_type not in ("P", "E"):
            raise ValueError("core_type must be 'P' or 'E'")
        if margin_mode not in ("none", "global", "gmm"):
            raise ValueError("margin_mode must be 'none', 'global' or 'gmm'")

        super().__init__(core_to_freq_mhz={c: base_frequency for c in range(config.system_cores)})

        self._core_type = core_type
        self._margin_mode = margin_mode
        self._global_margin = global_margin
        self._tcc_enabled = tcc_enabled
        self._tcc_temp = tcc_temp
        self._max_temperature = max_temperature
        self._step = frequency_step_size
        self._max_steps = max_steps
        p_range = p_core_range if p_core_range is not None else list(range(2500, 4500, 500))
        e_range = e_core_range if e_core_range is not None else list(range(1800, 3201, 200))
        self._freq_min, self._freq_max = (min(p_range), max(p_range)) if core_type == "P" else (min(e_range), max(e_range))

        # Models (full pickled modules, see module docstring)
        _register_model_classes()
        self._thermal_model = torch.load(thermal_model_path, map_location="cpu", weights_only=False)
        self._perf_model = torch.load(performance_model_path, map_location="cpu", weights_only=False)
        self._thermal_model.eval()
        self._perf_model.eval()
        self._thermal_model.forward = types.MethodType(_thermal_forward_with_embedding, self._thermal_model)

        # Scalers of the performance model
        self._input_scaler = joblib.load(os.path.join(deployment_dir, input_scaler_file))
        self._output_scaler = joblib.load(os.path.join(deployment_dir, output_scaler_file))

        # GMM-based uncertainty (only loaded when needed)
        self._gmm = None
        self._cluster_p90 = None
        if margin_mode == "gmm":
            self._gmm = joblib.load(os.path.join(deployment_dir, gmm_file))
            cluster_stat = joblib.load(os.path.join(deployment_dir, cluster_stat_file))
            self._cluster_p90 = np.array([info["p90"] for info in cluster_stat.values()])

        # Statistics
        self.number_of_thermal_violations = 0
        self.total_counter = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _clamp(self, freq: float) -> int:
        return int(min(max(freq, self._freq_min), self._freq_max))

    @staticmethod
    def _core_temperatures(system_state: SystemState, n: int) -> list[dict[int, float]]:
        # Public ARDiS exposes get_core_temperatures; older forks used get_core_temperature_metrics.
        buf = system_state.event_buffer
        getter = getattr(buf, "get_core_temperatures", None) or getattr(buf, "get_core_temperature_metrics")
        return getter(n)

    @staticmethod
    def _pid_of(system_state: SystemState, core: int) -> int:
        return next(system_state.app_to_pid[app] for app, cores in system_state.app_to_cores.items() if core in cores)

    def _build_feature_vector(self, system_state: SystemState, pid: int, core: int):
        """Features of the last _NUM_EPOCHS epochs for the performance and thermal models."""
        buf = system_state.event_buffer
        app_events_seq = buf.get_metrics_for_pid(pid, _NUM_EPOCHS)
        sys_events_seq = buf.get_system_metrics(_NUM_EPOCHS)
        freqs_seq = buf.get_core_frequencies(_NUM_EPOCHS)
        temps_seq = self._core_temperatures(system_state, _NUM_EPOCHS)

        assert len(app_events_seq) == _NUM_EPOCHS
        assert len(sys_events_seq) == _NUM_EPOCHS
        assert len(freqs_seq) == _NUM_EPOCHS

        perf_vec, thermal_vec = [], []
        current_instructions = 0
        for app, sys_ev, freqs, temps in zip(app_events_seq, sys_events_seq, freqs_seq, temps_seq):
            g = app.get
            freq = int(np.round(freqs.get(core, -2)))
            perf_vec.append([
                g("instructions", 0), g("branch-instructions", 0), g("LLC-loads", 0), g("LLC-stores", 0),
                g("branch-misses", 0), g("LLC-load-misses", 0), g("LLC-store-misses", 0), g("cycles", 0),
                g("cycle_activity.cycles_mem_any", 0), g("mem_inst_retired.any", 0),
                g("cycle_activity.stalls_l3_miss", 0), freq,
            ])
            thermal_vec.append([
                core, np.log1p(g("instructions", 0)), np.log1p(g("branch-instructions", 0)),
                np.log1p(g("LLC-loads", 0)), np.log1p(g("LLC-stores", 0)), np.log1p(g("branch-misses", 0)),
                np.log1p(g("LLC-load-misses", 0)), np.log1p(g("LLC-store-misses", 0)), np.log1p(g("cycles", 0)),
                freq / 100, sys_ev.get("power/energy-pkg/", 0), sys_ev.get("power/energy-cores/", 0),
                sys_ev.get("power/energy-psys/", 0),
            ] + list(temps.values()))
            current_instructions = g("instructions", 0)
        return perf_vec, thermal_vec, current_instructions

    def _forecast_temperature(self, thermal_vec, target_freq: float):
        """Thermal model forecast at target_freq. Returns (temps [1, n_cores], embedding)."""
        rows = [row + [int(target_freq / 100)] for row in thermal_vec]
        x = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)
        return self._thermal_model(x)

    def _forecast_instruction_diff(self, perf_vec, current_instructions, target_freq: float) -> float:
        """current_instructions - forecast instructions at target_freq."""
        rows = [row + [int(target_freq)] for row in perf_vec]
        x = torch.tensor(self._input_scaler.transform(rows), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self._perf_model(x).cpu().numpy().reshape(-1, 1)
        forecast = self._output_scaler.inverse_transform(out)[0][0]
        return float(current_instructions - forecast)

    def _margin(self, embedding: torch.Tensor) -> float:
        """Safety margin (degC) added to a forecasted temperature."""
        if self._margin_mode == "global":
            return float(self._global_margin)
        if self._margin_mode == "gmm":
            labels = self._gmm.predict(embedding.detach().cpu().numpy())
            return float(np.max(self._cluster_p90[labels]))
        return 0.0

    def _boostability(self, system_state, core, hot_core, current_temps, freq_now, target_freq):
        """(normalised instruction change) / (temperature change at the hottest core).

        Returns (boostability, temperature_reduction) or None if the core has no usable data.
        """
        pid = self._pid_of(system_state, core)
        perf_vec, thermal_vec, cur_instr = self._build_feature_vector(system_state, pid, core)
        forecast, _ = self._forecast_temperature(thermal_vec, target_freq)
        c = float(current_temps[hot_core] - forecast[-1][hot_core])
        if abs(c) < _EPS:
            c = _EPS if c >= 0 else -_EPS
        b = self._forecast_instruction_diff(perf_vec, cur_instr, target_freq) / cur_instr
        return b / c, c

    # ------------------------------------------------------------------ #
    # Policy entry point
    # ------------------------------------------------------------------ #
    def get_dvfs_actions(self, system_state: SystemState) -> list[DVFSAction]:
        self.total_counter += 1
        temps_now = self._core_temperatures(system_state, 1)[-1]
        t_max = max(temps_now.values())
        hot_core = max(temps_now, key=temps_now.get)
        freqs_now = system_state.event_buffer.get_core_frequencies(1)[-1]
        cores = list(system_state.occupied_cores)

        if t_max > self._max_temperature:
            self.number_of_thermal_violations += 1

        # Thermal cap emulation: throttle everything to the minimum frequency.
        if self._tcc_enabled and t_max > self._tcc_temp:
            log.info("TCC active (%.1f > %.1f): throttling cores %s", t_max, self._tcc_temp, cores)
            return [DVFSAction(core_id=c, frequency_mhz=self._freq_min) for c in cores]

        if t_max > self._max_temperature:
            return self._phase1_throttle(system_state, cores, hot_core, t_max, temps_now, freqs_now)

        actions = self._phase2_boost(system_state, cores, hot_core, temps_now, freqs_now)
        if actions:
            return actions
        return self._phase3_swap(system_state, cores, hot_core, temps_now, freqs_now)

    # ------------------------------------------------------------------ #
    # Phase 1: reduce temperature to a safe state
    # ------------------------------------------------------------------ #
    def _phase1_throttle(self, system_state, cores, hot_core, t_max, temps_now, freqs_now):
        best_core, best_boostability, best_c = -1, None, None
        for core in cores:
            f = freqs_now[core]
            res = self._boostability(system_state, core, hot_core, temps_now, f, f + self._step)
            if res is None:
                continue
            boostability, c = res
            if f > self._freq_min and (best_boostability is None or boostability < best_boostability):
                best_core, best_boostability, best_c = core, boostability, c

        if best_core == -1:
            log.info("Phase 1: every occupied core is already at the minimum frequency")
            return []

        steps = math.ceil((t_max - self._max_temperature) / best_c)
        delta = min(self._step * steps, self._step * self._max_steps)
        delta = min(abs(freqs_now[best_core] - self._freq_min), abs(delta))
        new_freq = self._clamp(freqs_now[best_core] - delta)
        return [DVFSAction(core_id=best_core, frequency_mhz=new_freq)]

    # ------------------------------------------------------------------ #
    # Phase 2: exploit thermal headroom
    # ------------------------------------------------------------------ #
    def _phase2_boost(self, system_state, cores, hot_core, temps_now, freqs_now):
        best_core, best_boostability = -1, None
        for core in cores:
            f = freqs_now[core]
            res = self._boostability(system_state, core, hot_core, temps_now, f, f + self._step)
            if res is None:
                continue
            boostability, _ = res
            if f < self._freq_max and (best_boostability is None or boostability > best_boostability):
                best_core, best_boostability = core, boostability

        if best_core == -1:
            return []

        f = freqs_now[best_core]
        _, thermal_vec, _ = self._build_feature_vector(system_state, self._pid_of(system_state, best_core), best_core)
        for levels in range(self._max_steps, 0, -1):
            proposed = f + self._step * levels
            if proposed > self._freq_max:
                continue
            forecast, emb = self._forecast_temperature(thermal_vec, proposed)
            if forecast.max().item() + self._margin(emb) < self._max_temperature:
                return [DVFSAction(core_id=best_core, frequency_mhz=self._clamp(proposed))]
        return []

    # ------------------------------------------------------------------ #
    # Phase 3: boost one core, throttle another
    # ------------------------------------------------------------------ #
    def _phase3_swap(self, system_state, cores, hot_core, temps_now, freqs_now):
        victim, target = -1, -1
        victim_b, target_b = None, None
        last_target_freq = None  # NOTE: see TODO below
        for core in cores:
            f = freqs_now[core]
            last_target_freq = f - self._step
            res = self._boostability(system_state, core, hot_core, temps_now, f, f - self._step)
            if res is None:
                continue
            boostability, _ = res
            if f > self._freq_min and (victim_b is None or boostability < victim_b):
                victim, victim_b = core, boostability
            if f < self._freq_max and (target_b is None or boostability > target_b):
                target, target_b = core, boostability

        if victim == -1 or target == -1 or victim == target:
            return []

        t_perf, t_thermal, t_instr = self._build_feature_vector(system_state, self._pid_of(system_state, target), target)
        v_perf, v_thermal, v_instr = self._build_feature_vector(system_state, self._pid_of(system_state, victim), victim)
        cur = torch.tensor(list(temps_now.values()), dtype=torch.float32).unsqueeze(0)

        victim_at_min, _ = self._forecast_temperature(v_thermal, self._freq_min)

        for t_freq in range(self._freq_max, int(freqs_now[target]), -self._step):
            target_fc, emb_t = self._forecast_temperature(t_thermal, t_freq)
            first_check = (cur + target_fc - victim_at_min).max().item() + self._margin(emb_t)
            if first_check >= self._max_temperature:
                continue

            for v_freq in range(int(freqs_now[victim]), self._freq_min, -self._step):
                victim_fc, emb_v = self._forecast_temperature(v_thermal, v_freq)
                second_check = (cur + target_fc - victim_fc).max().item() + self._margin(emb_v)
                if second_check >= self._max_temperature:
                    continue

                # TODO: the original code evaluates both performance forecasts at the last
                # core's (f - step) from the boostability loop instead of at t_freq / v_freq.
                # Kept for identical behaviour; confirm whether that is intended.
                diff_t = self._forecast_instruction_diff(t_perf, t_instr, last_target_freq)
                diff_v = self._forecast_instruction_diff(v_perf, v_instr, last_target_freq)
                if diff_t + diff_v > 0:
                    return [
                        DVFSAction(core_id=target, frequency_mhz=self._clamp(t_freq)),
                        DVFSAction(core_id=victim, frequency_mhz=self._clamp(v_freq)),
                    ]
                break  # same behaviour as original: stop scanning victim frequencies
        return []


def _thermal_forward_with_embedding(model_self, x):
    """Forward pass of the thermal LSTM returning (forecast, last-step LSTM embedding)."""
    with torch.no_grad():
        lstm_out, _ = model_self.lstm(x)
        embedding = lstm_out[:, -1, :]
        return model_self.fc(embedding), embedding
