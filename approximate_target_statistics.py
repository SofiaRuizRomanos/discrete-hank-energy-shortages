# calibrate_ss.py
import copy


# tqdm for nicer progress bars
from tqdm import tqdm

# --- Your modules (these must define hank_ss and return_statistics) ---

import warnings
from brute_force import brute_force_test, CalibrationState, BruteForceState

warnings.filterwarnings("error")


PARAM_NAMES = ["beta", "B", "ce_min", "E_s"]  # change as you like

static_ss = {
    "gamma_c": 0.5,  # à la HA
    "sigma_c": 0.1,  # same for HHs and Fs
    "alpha_c": 0.04,  # same for HHs and Fs
    "nu_n": 0.5,  # frisch # à la HA
    "theta_p": 10.0,
    "theta_w": 10.0,
    "Y": 1.0,
    "p_index": 1.0,
    "tax_total": 0.0,
}

mutable_ss = {
    ## parameters
    # grids
    "n_z": 7,  # 5-10
    "rho_z": 0.9,  # à la HA (tutorial 3 sj toolkit)
    "sd_z": 0.5,  # à la HA (tutorial 3 sj toolkit)
    "n_a": 150,  # 40 - 200
    "amin": 0,  # à la HA (tutorial 3 sj toolkit)
    "amax": 100,  # à la HA (tutorial 3 sj toolkit) and Pieroni 80 - 200
    # households
    # 'beta' : 0.99, # à la HA # if beta = 0.9 solve_steady_state error: Singular matrix
    "beta": 0.995,  # discount rate - percentage should be high 0.98 - 0.9995
    ## Exogenous
    "B": 5.5,  # total asset supply should be around 6 5-9 maybe
    "E_s": 0.126,  # energy supply - value 0.01 - 0.5
    "ce_min": 0.05,  # minimum consumption of energy - value 0 - 0.01
}

mutable_bounds = [
    ("n_z", (5, 10), True),  # 5-10
    ("rho_z", (0.85, 0.999), False),  # 0.85 - 0.999
    ("sd_z", (0.1, 1.0), False),  # 0
    ("n_a", (40, 200), True),  # 40 - 200
    ("amin", (0, 1), True),  # à la HA (tutorial 3 sj toolkit)
    ("amax", (80, 200), True),  # à la HA (tutorial 3
    ("beta", (0.98, 0.9995), False),
    ("B", (3.0, 12.0), False),  # was (5
    ("ce_min", (1e-5, 0.01), False),
    ("E_s", (0.01, 0.30), False),
]

test_mutable_bounds = [
    # ("n_z", (5, 7), True),  # 5-10
    # ("rho_z", (0.85, 0.999), False),  # 0.85 - 0.999
    # ("sd_z", (0.3, 1.0), False),  # 0
    # ("n_a", (100, 200), True),  # 40 - 200
    # ("amax", (100, 200), True),  # à la HA (tutorial 3
    # ("beta", (0.995, 0.9968947368421053), False),
    ("B", (4, 6), False, 10),  # was (5
    # ("ce_min", (0.02, 0.05), False, 4),
    ("E_s", (0.07, 0.2), False, 20),
    ("r", (0.001, 0.1), False, 5),
    ("p_e", (0.01, 2.1), False, 10),
    # ("w", (0.6, 1), False, 4),
    # ("Y", (0.6, 1), False, 4),
]

unknown_bounds = {"r": (0.001, 0.1), "p_e": (1, 8), "w": (0, 1), "Y": (0.5, 1.5)}

unknowns_ss = {"r": 0.003, "p_e": 2.01, "w": 0.6, "Y": 1.2}

# r 0.001 - 0.1
# p_e 1 - 8
# w 0 - 1
targets_ss = {"asset_mkt": 0.0, "energy_mkt": 0.0, "target_w": 0.0, "goods_mkt": 0}


def _compute_total_combinations(mutable_params, n_tests):
    """Compute the total number of combinations for the brute force search."""
    # Each mutable param will be tested over n_tests values
    # So total = n_tests ** (number of mutable params)
    total = 1
    for param in mutable_params:
        _, _, _, n_iters = param
        total *= n_iters
    return total


if __name__ == "__main__":
    n_tests = 10
    total = _compute_total_combinations(test_mutable_bounds, n_tests)
    print(f"Starting brute-force search: {total} total combinations")
    pbar = tqdm(total=total, desc="Brute tests")

    calibration_state = CalibrationState(
        calibration=copy.deepcopy({**static_ss, **mutable_ss}),
        unknowns=copy.deepcopy(unknowns_ss),
        targets=copy.deepcopy(targets_ss),
    )
    brute_force_state = BruteForceState()

    brute_force_test(
        brute_force_state=brute_force_state,
        calibration_state=calibration_state,
        mutable_params=test_mutable_bounds,
        n_tests=n_tests,
        pbar=pbar,
    )

    brute_force_state._save_cache()

    pbar.close()
    # finish progress line
    print()
