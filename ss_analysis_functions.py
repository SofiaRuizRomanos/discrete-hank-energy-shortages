import numpy as np
import matplotlib

# matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def get_lorenz_curve(income: np.ndarray, distribution: np.ndarray):

    if distribution.ndim != 1 or income.ndim != 1 or distribution.size != income.size:
        raise ValueError(
            "distribution and variable must be 1D arrays of the same length."
        )
    # if np.any(distribution < 0):
    #     raise ValueError("distribution must be nonnegative.")

    # order both arrays so that income is strictly increasing
    poor_to_rich_order = np.argsort(income, kind="mergesort")
    distribution = distribution[poor_to_rich_order]
    income = income[poor_to_rich_order]

    # compute total income and mass of hhs for percentages
    total_mass = distribution.sum()
    income_mass = income * distribution
    total_income = income_mass.sum()

    lorenz_x_data = np.concatenate(([0.0], np.cumsum(distribution) / total_mass))
    lorenz_y_data = np.concatenate(([0.0], np.cumsum(income_mass) / total_income))

    return lorenz_x_data, lorenz_y_data


def get_gini_coeff(income: np.ndarray, distribution: np.ndarray):
    if distribution.ndim != 1 or income.ndim != 1 or distribution.size != income.size:
        raise ValueError(
            "distribution and variable must be 1D arrays of the same length."
        )
    # if np.any(distribution < 0):
    #     raise ValueError("distribution must be nonnegative.")

    lorenz_x_data, lorenz_y_data = get_lorenz_curve(income, distribution)
    return float(1.0 - 2.0 * np.trapezoid(lorenz_y_data, lorenz_x_data))


def get_lorenz_curve(income: np.ndarray, distribution: np.ndarray):

    if distribution.shape != income.shape:
        raise ValueError("Distribution and variable must be of the same shape.")
    # if np.any(distribution < 0):
    #     raise ValueError("Distribution must be nonnegative.")

    # transform income and distribution into 1-dimension arrays
    income = np.ravel(income, order="C")
    distribution = np.ravel(distribution, order="C")

    # order both arrays so that income is strictly increasing
    poor_to_rich_order = np.argsort(income, kind="mergesort")
    distribution = distribution[poor_to_rich_order]
    income = income[poor_to_rich_order]

    # compute total income and mass of hhs for percentages
    total_mass = distribution.sum()
    income_mass = income * distribution
    total_income = income_mass.sum()

    lorenz_x_data = np.concatenate(([0.0], np.cumsum(distribution) / total_mass))
    lorenz_y_data = np.concatenate(([0.0], np.cumsum(income_mass) / total_income))

    return lorenz_x_data, lorenz_y_data


def get_gini_coeff(income: np.ndarray, distribution: np.ndarray):
    if distribution.size != income.size:
        raise ValueError("Distribution and variable must be of the same length.")
    if np.any(distribution < 0):
        raise ValueError("Distribution must be nonnegative.")

    lorenz_x_data, lorenz_y_data = get_lorenz_curve(income, distribution)
    return float(1.0 - 2.0 * np.trapezoid(lorenz_y_data, lorenz_x_data))


def find_quantiles(
    variable: np.ndarray,
    distribution: np.ndarray,
    steps: np.ndarray,
    linear_approach=False,
):

    if distribution.shape != variable.shape:
        raise ValueError("Distribution and variable must be of the same shape.")
    # if np.any(distribution < 0):
    #     raise ValueError("Distribution must be nonnegative.")
    if np.any((steps < 0) | (steps > 1)):
        raise ValueError("quantiles must be within [0, 1]")

    variable_long = np.ravel(variable, order="C")
    distribution_long = np.ravel(distribution, order="C")

    # normalizing the distribution just in case np.sum(distribution) != 1
    total_mass = distribution_long.sum()
    distribution_long = distribution_long / total_mass

    # ordering arrays in ascending y order
    ascending_mask = np.argsort(variable_long, kind="mergesort")
    variable_long = variable_long[ascending_mask]
    distribution_long = distribution_long[ascending_mask]

    cumulative_distribution = np.cumsum(distribution_long)

    if linear_approach == False:
        quantile_indexes = np.searchsorted(cumulative_distribution, steps, side="left")
        return variable_long[quantile_indexes]
    else:
        unique_cum_distrib, indices_cum_distrib = np.unique(
            cumulative_distribution, return_index=True
        )
        unique_variable_long = variable_long[indices_cum_distrib]
        return np.interp(steps, unique_cum_distrib, unique_variable_long)


def average_var_over_quantiles(
    variable: np.ndarray,
    distribution: np.ndarray,
    reference: np.ndarray,
    quantiles: np.ndarray,
):

    if (variable.shape != distribution.shape) or (variable.shape != reference.shape):
        raise ValueError(
            "variable, distribution, and reference must have the same shape."
        )
    if np.any(distribution < 0):
        raise ValueError("distribution must be nonnegative.")
    if not np.all(np.diff(quantiles) >= 0):
        raise ValueError("quantiles must be sorted in ascending order.")

    output = np.zeros(len(quantiles) + 1, dtype=float)

    # 1) flatten
    reference = np.ravel(reference).astype(float)
    variable = np.ravel(variable).astype(float)
    distribution = np.ravel(distribution).astype(float)

    # 2) sort by reference
    order = np.argsort(reference, kind="mergesort")
    reference = reference[order]
    variable = variable[order]
    distribution = distribution[order]

    # 3) loop bins: (-inf, q0], (q0, q1], ..., (q_{K-2}, q_{K-1}]
    for i in range(len(quantiles) + 1):
        if i == 0:
            mask = reference <= quantiles[i]
        elif i == (len(quantiles)):
            mask = reference > quantiles[-1]
        else:
            mask = (reference > quantiles[i - 1]) & (reference <= quantiles[i])

        distribution_quantile = distribution[mask]
        variable_quantile = variable[mask]
        total_mass = distribution_quantile.sum()

        # guard against empty bin or zero weight
        output[i] = (
            (distribution_quantile * variable_quantile).sum() / total_mass
            if total_mass > 0
            else np.nan
        )

    return output


def draw_figure_1(ss):
    ss_internals = ss.internals["household_decision"]

    # ----------------------------------------------------
    # subplot 1:  Lorenz curves of the income distribution

    ## getting data
    lorenz_x, lorenz_y = get_lorenz_curve(ss_internals["inc"], ss_internals["D"])

    Pieroni_lorenz_file_path = "/Users/sofia/Documents/Dauphine/Master Thesis/Pieroni files/Replication_files_EER/main/model_ext/Pieroni_Lorenz_curve.csv"
    lorenz_x_Pieroni, lorenz_y_Pieroni = np.loadtxt(
        Pieroni_lorenz_file_path, delimiter=",", unpack=True
    )
    germany_lorenz = np.cumsum(np.array([0, 8, 12, 17, 22, 40]) / 100)
    italy_lorenz = np.cumsum(np.array([0, 6, 12, 17, 23, 41]) / 100)

    x_ticks = np.arange(0, 120, 20) / 100
    diagonal = x_ticks

    colors = {
        "model": "#5D0B0B",
        "Pieroni": "#1f77b4",
        "germany": "#ff7f0e",
        "diag": "#000000",
        "italy": "#5dade2",
    }

    ## plot subplot
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(x_ticks, diagonal, color=colors["diag"], linewidth=1.0)
    ax.plot(
        x_ticks,
        germany_lorenz,
        color=colors["germany"],
        linewidth=1.2,
        label="Germany",
    )
    ax.plot(x_ticks, italy_lorenz, color=colors["italy"], linewidth=1.2, label="Italy")
    ax.plot(
        lorenz_x_Pieroni,
        lorenz_y_Pieroni,
        color=colors["Pieroni"],
        linewidth=1.0,
        label="Pieroni",
    )
    ax.plot(
        lorenz_x, lorenz_y, color=colors["model"], linewidth=1.0, label="Discrete model"
    )

    ax.set_xlabel("Cumulative share of households")
    ax.set_ylabel("Cumulative share of income")
    ax.grid(True)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc="upper left", frameon=True)

    # ----------------------------------------------------
    # subplot 2: average energy expenditure share across income percentile groups

    ## getting data
    income_percentiles = find_quantiles(
        ss_internals["inc"],
        ss_internals["D"],
        np.arange(0.1, 1.0, 0.1),
        linear_approach=True,
    )
    ce_over_inc = ss_internals["ce"] / ss_internals["inc"]
    ce_over_inc_percentiles = average_var_over_quantiles(
        ce_over_inc, ss_internals["D"], ss_internals["inc"], income_percentiles
    )

    Pieroni_ce_over_inc_percentiles = [
        0.1177,
        0.0960,
        0.0897,
        0.0878,
        0.0864,
        0.0852,
        0.0844,
        0.0836,
        0.0830,
        0.0822,
    ]

    ## plot subplot
    ax = axes[1]

    x_ticks = np.arange(5, 105, 10)
    ax.plot(
        x_ticks,
        ce_over_inc_percentiles,
        color=colors["model"],
        linewidth=1.2,
        label="Discrete model",
    )
    ax.plot(
        x_ticks,
        Pieroni_ce_over_inc_percentiles,
        color=colors["Pieroni"],
        linewidth=1.2,
        label="Pieroni",
    )

    ax.set_xlabel("Income percentiles")
    ax.set_ylabel("Expenditure share (%)")
    ax.set_xlim(10, 100)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper right", frameon=True)

    return fig


def draw_figure_2(ss):

    ss_internals = ss.internals["household_decision"]

    # ----------------------------------------------------
    # subplot 1: average energy expenditure share across income quintile groups
    income_quintiles = find_quantiles(
        ss_internals["inc"],
        ss_internals["D"],
        np.arange(0.2, 1.0, 0.2),
        linear_approach=True,
    )
    ce_over_inc = ss_internals["ce"] / ss_internals["inc"]
    ce_over_inc_quintiles = average_var_over_quantiles(
        ce_over_inc, ss_internals["D"], ss_internals["inc"], income_quintiles
    )

    Pieroni_ce_over_inc_quintiles = [
        0.1049,
        0.0888,
        0.0858,
        0.0840,
        0.0826,
    ]  # from the Pieroni matlab code
    germany_ce_over_inc_quintiles = (
        2 / 100.0 * np.array([4.9, 4.6, 4.6, 4.2, 3.5])
    )  # from the Pieroni matlab code
    # we divide by 100 instead of multiplying the other data by 100 in the plot line like Pieroni does

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    ax = axes[0]

    colors = {
        "Pieroni": "#1f77b4",
        "model": "#975151",
        "bar_edge": "#593030",
        "germany": "#ff7f0e",
    }

    ax.bar(
        np.arange(1, 6, 1),
        ce_over_inc_quintiles.squeeze(),
        color=colors["model"],
        edgecolor=colors["bar_edge"],
        label="Discrete model",
    )
    ax.plot(
        np.arange(1, 6, 1),
        Pieroni_ce_over_inc_quintiles,
        linestyle="none",
        marker="o",
        markersize=6,
        color=colors["Pieroni"],
        label="Pieroni",
    )
    ax.plot(
        np.arange(1, 6, 1),
        germany_ce_over_inc_quintiles,
        linestyle="none",
        marker="o",
        markersize=6,
        color=colors["germany"],
        label="Germany",
    )

    ax.set_xlabel("Income quintiles")
    ax.set_ylabel("Expenditure share (%)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # ----------------------------------------------------
    # subplot 2: marginal propensities to consume

    ax = axes[1]
    Pieroni_mpc_inc_percentiles = [
        0.3910,
        0.3113,
        0.1484,
        0.1589,
        0.1023,
        0.0903,
        0.0788,
        0.0720,
        0.0476,
        0.0296,
    ]  # from the Pieroni matlab code

    income_percentiles = find_quantiles(
        ss_internals["inc"],
        ss_internals["D"],
        np.arange(0.1, 1.0, 0.1),
        linear_approach=True,
    )
    mpc_inc_percentiles = average_var_over_quantiles(
        ss_internals["mpc"], ss_internals["D"], ss_internals["inc"], income_percentiles
    )

    x_ticks = np.arange(5, 105, 10)

    ax.plot(
        x_ticks,
        Pieroni_mpc_inc_percentiles,
        marker="o",
        markersize=6,
        color=colors["Pieroni"],
        label="Pieroni",
    )
    ax.plot(
        x_ticks,
        mpc_inc_percentiles,
        marker="o",
        markersize=6,
        color=colors["model"],
        label="Discrete model",
    )

    ax.set_xlabel("Income percentiles")
    ax.set_ylabel("MPC (%)")
    ax.set_xlim(10, 100)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.legend(handles, labels, loc="upper right", frameon=True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    return fig


def return_statistics(ss):

    ss_internals = ss.internals["household_decision"]

    results_dict = {
        "average_wealth_to_income_share": {
            "Model": np.sum(
                ss_internals["D"]
                * (
                    ss_internals["a_grid"][np.newaxis, :]
                    / ss_internals["z_grid"][:, np.newaxis]
                )
            )
            / np.sum(ss_internals["D"]),
            "Pieroni": 4.4178,
            "Target": 4.2,
        },
        "total_energy_share_output": {
            "Model": (ss["p_e"] * ss["E_s"]) / (4 * ss["Y"]),
            "Pieroni": 0.0403,
            "Target": 0.04,
        },
        "average_energy_expenditure_share": {
            "Model": np.sum(
                ss_internals["D"] * (ss_internals["ce"] / ss_internals["c"])
            )
            / np.sum(ss_internals["D"]),
            "Pieroni": 0.0892,
            "Target": "[0.06:0.12]",
        },
        "gini": {
            "Model": get_gini_coeff(ss_internals["inc"], ss_internals["D"]),
            "Pieroni": 0.4812,
            "Target": 0.35,
        },
        "average_mpc": {
            "Model": np.sum(ss_internals["D"] * ss_internals["mpc"])
            / np.sum(ss_internals["D"]),
            "Pieroni": 0.1069,
            "Target": "[0.15:0.25]",
        },
        "annualized_percent_rate": {
            "Model": ss["r"] * 4 * 100,
            "Pieroni": 2.9139,
        },
    }

    # def fmt(x):
    #     try:
    #         return f"{float(x):.4f}"
    #     except Exception:
    #         return str(x)

    # for stat_name, vals in results_dict.items():
    #     print(
    #         f"Statistic: {stat_name}\n"
    #         f"  Model: {fmt(vals.get('Model'))}\n"
    #         f"  Pieroni: {fmt(vals.get('Pieroni'))}\n"
    #         f"  Target: {fmt(vals.get('Target'))}\n"
    #     )
    return results_dict


def draw_figures(ss):
    ss_internals = ss.internals["household_decision"]

    # ----------------------------------------------------
    # subplot 1:  Lorenz curves of the income distribution

    ## getting data
    lorenz_x, lorenz_y = get_lorenz_curve(ss_internals["inc"], ss_internals["D"])

    Pieroni_lorenz_file_path = "./Pieroni_Lorenz_curve.csv"
    lorenz_x_Pieroni, lorenz_y_Pieroni = np.loadtxt(
        Pieroni_lorenz_file_path, delimiter=",", unpack=True
    )
    germany_lorenz = np.cumsum(np.array([0, 8, 12, 17, 22, 40]) / 100)
    italy_lorenz = np.cumsum(np.array([0, 6, 12, 17, 23, 41]) / 100)

    x_ticks = np.arange(0, 120, 20) / 100
    diagonal = x_ticks

    colors = {
        "model": "#5D0B0B",
        "Pieroni": "#1f77b4",
        "germany": "#ff7f0e",
        "diag": "#000000",
        "italy": "#5dade2",
    }

    ## plot subplot
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x_ticks, diagonal, color=colors["diag"], linewidth=1.0)
    ax.plot(
        x_ticks,
        germany_lorenz,
        color=colors["germany"],
        linewidth=1.2,
        label="Germany",
    )
    ax.plot(x_ticks, italy_lorenz, color=colors["italy"], linewidth=1.2, label="Italy")
    ax.plot(
        lorenz_x_Pieroni,
        lorenz_y_Pieroni,
        color=colors["Pieroni"],
        linewidth=1.0,
        label="Pieroni",
    )
    ax.plot(
        lorenz_x, lorenz_y, color=colors["model"], linewidth=1.0, label="Discrete model"
    )

    ax.set_title("Lorenz Curve of household income")
    ax.set_xlabel("Cumulative share of households")
    ax.set_ylabel("Cumulative share of income")
    ax.grid(True)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc="upper left", frameon=True)

    # ----------------------------------------------------
    # subplot 2: average energy expenditure share of income earned across income percentile groups

    ## getting data
    income_percentiles = find_quantiles(
        ss_internals["inc"],
        ss_internals["D"],
        np.arange(0.1, 1.0, 0.1),
        linear_approach=True,
    )
    ce_over_inc = ss_internals["ce"] / ss_internals["inc"]
    ce_over_inc_percentiles = average_var_over_quantiles(
        ce_over_inc, ss_internals["D"], ss_internals["inc"], income_percentiles
    )

    Pieroni_ce_over_inc_percentiles = [
        0.1177,
        0.0960,
        0.0897,
        0.0878,
        0.0864,
        0.0852,
        0.0844,
        0.0836,
        0.0830,
        0.0822,
    ]

    ## plot subplot
    ax = axes[0, 1]

    x_ticks = np.arange(5, 105, 10)
    ax.plot(
        x_ticks,
        ce_over_inc_percentiles,
        color=colors["model"],
        linewidth=1.2,
        label="Discrete model",
    )
    ax.plot(
        x_ticks,
        Pieroni_ce_over_inc_percentiles,
        color=colors["Pieroni"],
        linewidth=1.2,
        label="Pieroni",
    )

    ax.set_title("Share of energy expenditure over income")
    ax.set_xlabel("Income percentiles")
    ax.set_ylabel("Expenditure share (%)")
    ax.set_xlim(10, 100)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper right", frameon=True)

    # ----------------------------------------------------
    # subplot 3: average energy expenditure share of income earned across income quintile groups

    ## getting data
    income_quintiles = find_quantiles(
        ss_internals["inc"],
        ss_internals["D"],
        np.arange(0.2, 1.0, 0.2),
        linear_approach=True,
    )
    ce_over_inc_quintiles = average_var_over_quantiles(
        ce_over_inc, ss_internals["D"], ss_internals["inc"], income_quintiles
    )

    # from the Pieroni matlab code:
    Pieroni_ce_over_inc_quintiles = [
        0.1049,
        0.0888,
        0.0858,
        0.0840,
        0.0826,
    ]
    germany_ce_over_inc_quintiles = (
        2 / 100.0 * np.array([4.9, 4.6, 4.6, 4.2, 3.5])
    )  # we divide by 100 instead of multiplying the other data by 100 in the plot line like Pieroni does

    ## plot subplot
    ax = axes[1, 0]

    colors = {
        "Pieroni": "#1f77b4",
        "model": "#975151",
        "bar_edge": "#593030",
        "germany": "#ff7f0e",
    }

    ax.bar(
        np.arange(1, 6, 1),
        ce_over_inc_quintiles.squeeze(),
        color=colors["model"],
        edgecolor=colors["bar_edge"],
        label="Discrete model",
    )
    ax.plot(
        np.arange(1, 6, 1),
        Pieroni_ce_over_inc_quintiles,
        linestyle="none",
        marker="o",
        markersize=6,
        color=colors["Pieroni"],
        label="Pieroni",
    )
    ax.plot(
        np.arange(1, 6, 1),
        germany_ce_over_inc_quintiles,
        linestyle="none",
        marker="o",
        markersize=6,
        color=colors["germany"],
        label="Germany",
    )

    ax.set_title("Share of energy expenditure over income")
    ax.set_xlabel("Income quintiles")
    ax.set_ylabel("Expenditure share (%)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # ----------------------------------------------------
    # subplot 4: marginal propensities to consume

    ax = axes[1, 1]
    Pieroni_mpc_inc_percentiles = [
        0.3910,
        0.3113,
        0.1484,
        0.1589,
        0.1023,
        0.0903,
        0.0788,
        0.0720,
        0.0476,
        0.0296,
    ]  # from the Pieroni matlab code

    income_percentiles = find_quantiles(
        ss_internals["inc"],
        ss_internals["D"],
        np.arange(0.1, 1.0, 0.1),
        linear_approach=True,
    )
    mpc_inc_percentiles = average_var_over_quantiles(
        ss_internals["mpc"], ss_internals["D"], ss_internals["inc"], income_percentiles
    )

    x_ticks = np.arange(5, 105, 10)

    ax.plot(
        x_ticks,
        Pieroni_mpc_inc_percentiles,
        marker="o",
        markersize=6,
        color=colors["Pieroni"],
        label="Pieroni",
    )
    ax.plot(
        x_ticks,
        mpc_inc_percentiles,
        marker="o",
        markersize=6,
        color=colors["model"],
        label="Discrete model",
    )

    ax.set_title("Marginal propensity to consume")
    ax.set_xlabel("Income percentiles")
    ax.set_ylabel("MPC (%)")
    ax.set_xlim(10, 100)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.legend(handles, labels, loc="upper right", frameon=True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    return fig
