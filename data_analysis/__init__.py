"""Package data_analysis : calibration empirique et comparaison avec données réelles."""

from data_analysis.reward_calibration import (
    calibrate_penalties_from_data,
    compute_trajectory_metrics,
    generate_human_reference_trajectory,
    get_calibration_report,
)

__all__ = [
    "calibrate_penalties_from_data",
    "compute_trajectory_metrics",
    "generate_human_reference_trajectory",
    "get_calibration_report",
]
