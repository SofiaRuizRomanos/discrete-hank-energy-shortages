import sys
import model_functions as hank  # must define hank_ss and blocks
from matplotlib import pyplot as plt
import ss_analysis_functions as analysis  # must define return_statistics
from math import isfinite
import pickle
import os

hank_ss = hank.build_hank_ss_model()


class BruteForceState:
    def __init__(self):
        self.global_total_solutions = 0
        self.global_best_score = float("inf")
        self._load_cache()

    def _save_cache(self):

        with open("brute_force_test_cache.pkl", "wb") as f:
            pickle.dump(self.test_cache, f)

    def _load_cache(self):
        import pickle

        if os.path.exists("brute_force_test_cache.pkl"):
            with open("brute_force_test_cache.pkl", "rb") as f:
                self.test_cache = pickle.load(f)
        else:
            self.test_cache = {}

    def get_best_from_cache(self):
        best_score = float("inf")
        best_ss = None
        best_params = None
        for calibration, ss in self.test_cache.items():
            if ss == "ERROR":
                continue
            try:
                stats = analysis.return_statistics(ss)
                score = score_stats(stats)
                if score < best_score:
                    best_score = score
                    best_ss = ss
                    best_params = calibration
            except:
                continue
        return best_ss, best_score, best_params


class CalibrationState:
    def __init__(self, calibration: dict, unknowns: dict, targets: dict):
        self.calibration = calibration
        self.unknowns = unknowns
        self.targets = targets

    def set(self, name, value):
        if name in self.calibration:
            self.calibration[name] = value
        elif name in self.unknowns:
            self.unknowns[name] = value
        else:
            raise AttributeError(f"{name} not found in calibration or unknowns.")

    def get(self, name):
        if name in self.calibration:
            return self.calibration[name]
        elif name in self.unknowns:
            return self.unknowns[name]
        else:
            raise AttributeError(f"{name} not found in calibration or unknowns.")

    def __str__(self):
        return str({**self.calibration, **self.unknowns}.items())


model_targets = {
    # "name": (target_value, multiplier, (low_bound, high_bound))
    # multiplier is used to weight the importance of each statistic in the scoring function
    # Currently standardized so that each reaches a value of 10
    "average_wealth_to_income_share": (4.2, 2.38, (4, 4.4)),
    "total_energy_share_output": (0.04, 250, (0.04, 0.043)),
    "average_energy_expenditure_share": (0.09, 111.11, (0.06, 0.12)),
    "gini": (0.35, 28.57, (0.3, 0.5)),  # sofia added
}


def score_stats(stats):
    """Return a score based on how close the statistics are to target values."""
    targets = model_targets
    score = 0.0
    for stat_name, target_tup in targets.items():
        target, multiplier, (low, high) = target_tup
        model_value = stats.get(stat_name, {}).get("Model", None)
        if model_value is None or not isfinite(model_value):
            return float("inf")  # Penalize missing or non-finite values heavily
        if low <= model_value <= high:
            continue
        score += abs(model_value - target) * multiplier
    return score


def check_stats(stats):

    targets = {param_name: values[2] for param_name, values in model_targets.items()}

    for stat_name, (low, high) in targets.items():
        model_value = stats.get(stat_name, {}).get("Model", None)
        if model_value is None or not (low <= model_value <= high):
            return False
    return True


def check_cache(
    calibration_state: CalibrationState, brute_force_state: BruteForceState
):
    cache_key = str(calibration_state)
    if cache_key in brute_force_state.test_cache:
        ss = brute_force_state.test_cache[cache_key]
        if ss == "ERROR":
            raise Exception("Previously failed calibration.")
        return ss
    return None


def solve_and_cache(
    calibration_state: CalibrationState, brute_force_state: BruteForceState
):
    cache_key = str(calibration_state)
    try:
        ss = hank_ss.solve_steady_state(
            calibration_state.calibration,
            calibration_state.unknowns,
            calibration_state.targets,
            options={"verbose": False, "ttol": 1e-6},
            solver_kwargs={"maxcount": 5000},
        )
        brute_force_state.test_cache[cache_key] = ss
        return ss

    except Exception as e:
        brute_force_state.test_cache[cache_key] = "ERROR"
        raise e


def run_test(calibration_state: CalibrationState, brute_force_state: BruteForceState):

    ss = check_cache(calibration_state, brute_force_state)
    if ss is None:
        ss = solve_and_cache(calibration_state, brute_force_state)
    return ss


def write_solution_files(folder, stats, calibration_state, brute_force_state, ss):
    with open(
        f"{folder}/stats/solution_{brute_force_state.global_total_solutions}_stats.txt",
        "w",
    ) as f:
        f.write("=== STEADY STATE STATISTICS ===\n")
        for stat_name, stat_val in stats.items():
            f.write(f"{stat_name}: \n")
            for author, val in stat_val.items():
                f.write(f"  {author}: {val}\n")
    with open(
        f"{folder}/calibration/solution_{brute_force_state.global_total_solutions}_calibration.txt",
        "w",
    ) as f:
        f.write("calibration_ss = {\n")
        for param, val in {
            **calibration_state.calibration,
        }.items():
            f.write(f"'{param}': {val},\n")
        f.write("}")
        f.write("\nunknowns_ss = {\n")
        for param, val in {
            **calibration_state.unknowns,
        }.items():
            f.write(f"'{param}': {val},\n")
        f.write("}\n")

    with open(
        f"{folder}/graphs/solution_{brute_force_state.global_total_solutions}.png",
        "wb",
    ) as f:
        analysis.draw_figures(ss)
        outpath = (
            f"{folder}/graphs/solution_{brute_force_state.global_total_solutions}.png"
        )
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()  # closes the current figure


def store_solution(
    calibration_state: CalibrationState,
    brute_force_state: BruteForceState,
    stats,
    improvement,
    ss,
):
    folder = "rejected_solutions"

    if check_stats(stats):
        folder = "solutions"
        print(f"\nFound solution #{brute_force_state.global_total_solutions}")
    if improvement:
        folder = "best_solutions"
        print(
            f"\nNew best score with solution #{brute_force_state.global_total_solutions}: {brute_force_state.global_best_score:.6g} "  # output the fact that we found something better
        )

    if folder != "rejected_solutions":  # uncomment to save all solutions
        write_solution_files(
            f"brute_force_results/{folder}",
            stats,
            calibration_state,
            brute_force_state,
            ss,
        )


def brute_force_test(
    calibration_state: CalibrationState,
    mutable_params,
    brute_force_state: BruteForceState,
    n_tests=10,
    pbar=None,
):
    if mutable_params:
        param_to_vary = mutable_params[0]
        param_name, (low, high), is_integer, n_iters = param_to_vary
        for i in range(n_iters):
            val = low + (high - low) * i / (n_iters - 1)
            if is_integer:
                val = int(round(val))

            calibration_state.set(param_name, val)

            brute_force_test(
                calibration_state,
                mutable_params[1:],
                brute_force_state,
                n_tests,
                pbar,
            )
    else:
        try:

            ss = run_test(calibration_state, brute_force_state)
            stats = analysis.return_statistics(ss)
            score = score_stats(stats)
            improvement = False
            if score < brute_force_state.global_best_score:
                improvement = True
                brute_force_state.global_best_score = score
            brute_force_state.global_total_solutions += 1
            store_solution(calibration_state, brute_force_state, stats, improvement, ss)
            pbar.update(1)

        except Exception as e:
            pbar.update(1)


if __name__ == "__main__":
    brute_force_state = BruteForceState()
    best_ss, best_score, best_params = brute_force_state.get_best_from_cache()
    print(f"Best score from cache: {best_score} with params {best_params}")
    print(analysis.return_statistics(best_ss))
