"""SmartBoost-HW testing: run every policy variant against your trained models.

Expected layout (the models/ folder next to this file; see models/README.md):

    SmartBoost-HW/models/
        thermal_model_lstm.pth
        performance_model_lstm_1_app.pth
        input_scaler.pkl
        output_scaler.pkl
        gmm_thermal_model.pkl            (only for margin_mode="gmm")
        cluster_stat_thermal.pkl         (only for margin_mode="gmm")

The model architectures (thermal_model.py / perfromance_model.py) sit next to
smartboost_hw.py and are registered by the policy, so nothing has to be imported here.

Usage (from the repo root):
    python SmartBoost-HW/smartboost_hw_testing.py                 # list variants
    python SmartBoost-HW/smartboost_hw_testing.py E_gmm           # run one variant
    python SmartBoost-HW/smartboost_hw_testing.py all             # run all of them
"""

import os
import sys

import ardis.config as config
from ardis.benchmarks import ParsecApplication
from ardis.core import MonitoringMode
from ardis.core.policies import ConsecutiveScheduler, ExplicitMapping, SmartBoostHW
from ardis.core.postprocessing import SimpleClipPostProcessor
from ardis.core.postprocessing.simple_clip_postprocessor import Clips
from ardis.experiments import Experiment

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
THERMAL_MODEL = os.path.join(MODELS_DIR, "thermal_model_lstm.pth")
PERFORMANCE_MODEL = os.path.join(MODELS_DIR, "performance_model_lstm_1_app.pth")

# name -> SmartBoostHW options. Each entry replaces one of the old SmartBoost_dvfs_* files.
VARIANTS = {
    "E_none":      dict(core_type="E", margin_mode="none"),
    "E_global":    dict(core_type="E", margin_mode="global", global_margin=1.7254733),
    "E_gmm":       dict(core_type="E", margin_mode="gmm"),
    "E_gmm_tcc":   dict(core_type="E", margin_mode="gmm", tcc_enabled=True, tcc_temp=41.5),
    "P_none":      dict(core_type="P", margin_mode="none"),
    "P_global":    dict(core_type="P", margin_mode="global", global_margin=1.7254733),
    "P_gmm":       dict(core_type="P", margin_mode="gmm"),
    "P_gmm_tcc":   dict(core_type="P", margin_mode="gmm", tcc_enabled=True, tcc_temp=41.5),
}


def run_variant(name: str) -> dict:
    opts = VARIANTS[name]
    policy = SmartBoostHW(
        thermal_model_path=THERMAL_MODEL,
        performance_model_path=PERFORMANCE_MODEL,
        deployment_dir=MODELS_DIR,
        max_temperature=40,
        frequency_step_size=200,
        max_steps=5,
        p_core_range=list(range(1000, 4501, 100)),
        e_core_range=list(range(1000, 3501, 100)),
        base_frequency=1000,
        **opts,
    )

    core_ids = config.intel_p_core_ids if opts["core_type"] == "P" else config.intel_e_core_ids
    exp = Experiment(
        name=f"smartboost_hw_test_{name}",
        mapping_policy=ExplicitMapping.from_list(core_ids[:8]),
        scheduler=ConsecutiveScheduler(delay_sec=0),
        dvfs_policy=policy,
        applications=[
            ParsecApplication("parsec.blackscholes"),
            ParsecApplication("splash2x.radix"),
            ParsecApplication("parsec.dedup"),
            ParsecApplication("parsec.canneal"),
        ],
        monitoring_mode=MonitoringMode.PERIODIC_ON_PID,
        postprocessor=SimpleClipPostProcessor(
            clips=[
                Clips.APP_METRICS_ALL,
                Clips.THREAD_EXECUTION_OVERVIEW, Clips.THREAD_MAPPING,
                Clips.SYSTEM_METRICS_ALL,
                Clips.SYSTEM_CORE_FREQUENCY,
                Clips.TEMPERATURE_PLOT,
            ],
            verbose=True,
        ),
    )
    exp.executeExperiment()

    result = {
        "variant": name,
        "thermal_violations": policy.number_of_thermal_violations,
        "epochs": policy.total_counter,
    }
    print(result)
    return result


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        print("variants:", ", ".join(VARIANTS), "| or 'all'")
    elif arg == "all":
        for variant in VARIANTS:
            run_variant(variant)
    elif arg in VARIANTS:
        run_variant(arg)
    else:
        sys.exit(f"unknown variant '{arg}'. choose from: {', '.join(VARIANTS)}, all")
