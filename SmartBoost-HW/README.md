# SmartBoost-HW for ARDiS

SmartBoost-HW is a model-based DVFS policy (SmartBoost on real hardware) for
[ARDiS](https://github.com/Chair-for-Embedded-Systems/ARDiS). It keeps the chip below a
temperature limit while maximising performance, using two learned forecasters:

- a **thermal model** that predicts the per-core temperatures at a candidate frequency, and
- a **performance model** that predicts the instruction throughput at a candidate frequency.

From the two it computes a *boostability* per core, (relative instruction change) / (temperature
change at the hottest core), and works in three phases:

1. **Throttle:** the chip is above `max_temperature`, so lower the core with the lowest boostability.
2. **Boost:** there is thermal headroom, so raise the core with the highest boostability.
3. **Swap:** there is no headroom, so raise one core and lower another when the net result is a gain.

> **Status:** work in progress. The policy has not yet been validated in this repository layout.

## What is in this repository

```
ardis/core/policies/dvfs/
    smartboost_hw.py         the policy (one configurable class, SmartBoostHW)
    thermal_model.py         thermal model architecture      (PLACEHOLDER, see below)
    performance_model.py     performance model architecture  (PLACEHOLDER, see below)
SmartBoost-HW/
    smartboost_hw_testing.py run SmartBoost-HW variants on real hardware
    models/                  put your trained models here (see models/README.md)
```

The trained models and scalers are **not** included.

## Setup

1. Install ARDiS as described in its README (including PARSEC), and `torch`, `joblib` and `scikit-learn`.
2. Register the policy in `ardis/core/policies/__init__.py`:
   ```python
   from .dvfs.smartboost_hw import SmartBoostHW      # inside the TYPE_CHECKING block
   'SmartBoostHW',                                    # in __all__
   'SmartBoostHW': '.dvfs.smartboost_hw',             # in _LAZY_IMPORTS
   ```
3. **Fill in the model placeholders.** In `thermal_model.py` and `performance_model.py` every
   number of the architecture was replaced by `N`. Replace each `N` with the value of your own
   trained model (LSTM hidden sizes, number of layers, feature count, output size, dense
   widths). The comment next to each `N` says what it is. The files raise a clear error while
   any `N` is left.

   The architecture has to be exactly the one your `.pth` files were saved with.
4. Copy your trained models and scalers into `SmartBoost-HW/models/`.

## Run

From the repository root, with the privileges you normally use for ARDiS:

```
python SmartBoost-HW/smartboost_hw_testing.py             # list the variants
python SmartBoost-HW/smartboost_hw_testing.py E_gmm       # run one
python SmartBoost-HW/smartboost_hw_testing.py all         # run all
```

Use it in your own experiment like any ARDiS DVFS policy:

```python
from ardis.core.policies import SmartBoostHW

policy = SmartBoostHW(
    thermal_model_path="SmartBoost-HW/models/thermal_model.pth",
    performance_model_path="SmartBoost-HW/models/performance_model.pth",
    deployment_dir="SmartBoost-HW/models",
    core_type="E",          # "P" or "E": one core type per experiment
    margin_mode="gmm",      # "none" | "global" | "gmm"
    max_temperature=80,
)
# Experiment(..., dvfs_policy=policy, ...)
```

## Options

| Option | Values | Meaning |
|---|---|---|
| `core_type` | `"P"`, `"E"` | Frequency range and cores the policy controls |
| `margin_mode` | `"none"` | Trust the thermal model as is |
| | `"global"` | Add a fixed safety margin (`global_margin`, in degC) |
| | `"gmm"` | Add a per-window uncertainty margin: the `p90` of the GMM cluster of the thermal-model embedding |
| `tcc_enabled`, `tcc_temp` | bool, degC | Thermal-cap emulation: above `tcc_temp`, throttle all occupied cores to the minimum frequency |
| `max_temperature` | degC | Temperature limit the policy tries to stay under |
| `frequency_step_size`, `max_steps` | MHz, int | Step size and the maximum number of steps per decision |
| `p_core_range`, `e_core_range` | list of MHz | Allowed frequencies per core type |

The variants in `smartboost_hw_testing.py` (`E_none`, `E_global`, `E_gmm`, `E_gmm_tcc` and the same
for `P_`) are combinations of these options.

## Notes

- The policy needs `get_core_temperatures` (or `get_core_temperature_metrics`) on the ARDiS event buffer.
- `margin_mode="gmm"` needs `gmm.pkl` and `cluster_stats.pkl`, produced from the
  thermal-model embeddings of your training data.
- The model files are pickled `torch` modules loaded with `weights_only=False`. Only load files you trust.
