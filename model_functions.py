# Importing
import numpy as np
import sequence_jacobian as ssj


### Het inputs


def make_grids(rho_z, sd_z, n_z, amin, amax, n_a):
    z_grid, pi_stationary, Pi_trans = ssj.grids.markov_rouwenhorst(
        rho=rho_z, sigma=sd_z, N=n_z
    )
    a_grid = ssj.grids.agrid(amin=amin, amax=amax, n=n_a)
    # check : sum of probabilities for all states of productivity should be equal to 1
    check_sum_productivity = sum(pi_stationary)
    return z_grid, pi_stationary, Pi_trans, a_grid, check_sum_productivity


def set_wages(w, z_grid):
    wz = w * z_grid
    return wz


def set_transfers(profit, tax_total, z_grid, pi_stationary):
    # hardwired incidence rules are proportional to skill; scale does not matter
    # tax collection and dividend distribution are respectively proportional and inversely proportional to productivity
    div = z_grid * (
        profit / np.sum(pi_stationary * z_grid)
    )  # the higher the productivity, the higher the dividend
    tax = z_grid * (
        tax_total / np.sum(pi_stationary * z_grid)
    )  # the higher the productivity, the higher the tax
    transfers = div - tax
    return transfers, div, tax


# Household Block
def hh_initial_guess_previous(a_grid, wz, transfers, r, gamma_c):
    ## classic first guess:
    coh = (1 + r) * a_grid[np.newaxis, :] + wz[:, np.newaxis] + transfers[:, np.newaxis]
    Vprime_a = (1 + r) * (0.1 * coh) ** (
        -gamma_c
    )  # consumption is 10% of cash on hands, assets are 90%

    return Vprime_a


# Household Block
def hh_initial_guess(a_grid, wz, transfers, r, gamma_c):
    coh = (1 + r) * a_grid[np.newaxis, :] + wz[:, np.newaxis] + transfers[:, np.newaxis]
    Vprime_a = (1 + r) * (0.1 * coh) ** (-gamma_c)
    return Vprime_a


@ssj.het(
    exogenous="Pi_trans",
    policy="a",
    backward="Vprime_a",
    backward_init=hh_initial_guess,
)
def household_decision(Vprime_a_p, a_grid, z_grid, wz, transfers, r, beta, gamma_c, N):
    # 1. Endog Gridpoint Method: Get c on the next asset grid
    Uprime_c_nextgrid = beta * Vprime_a_p
    c_nextgrid = Uprime_c_nextgrid ** (-1 / gamma_c)

    # 2. Interpolate to get the values of c on the current period asset grid
    coh = (
        (1 + r) * a_grid[np.newaxis, :]
        + wz[:, np.newaxis] * N
        + transfers[:, np.newaxis]
    )
    a = ssj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)

    ssj.misc.setmin(a, a_grid[0])  # Cannot borrow more than a_grid[0]
    c = coh - a

    # Total resources and effective labor of the agent
    inc = wz[:, np.newaxis] * N + (1 + r) * a + transfers[:, np.newaxis]
    nz = z_grid[:, np.newaxis] * np.ones_like(c) * N

    # Update the marginal value function
    Vprime_a = (1 + r) * c ** (-gamma_c)

    return Vprime_a, a, c, inc, nz


def intratemporal_consumption(ce_min, alpha_c, sigma_c, p_e, p_g, p_index, c):
    ce = ce_min + alpha_c * c * ((p_e / p_index) ** (-sigma_c))
    cg = (1 - alpha_c) * c * ((p_g / p_index) ** (-sigma_c))
    return ce, cg


def get_mpc_transfers(c, a, a_grid, r, transfers):
    """Approximate mpc, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * a_grid[np.newaxis, :] + transfers[:, np.newaxis]
    mpc[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[:, 2:] - post_return[:, :-2])
    mpc[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[:, 1] - post_return[:, 0])
    mpc[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[:, -1] - post_return[:, -2])
    mpc[a == a_grid[0]] = 1
    return mpc


# Simple functions


@ssj.simple
def firm(alpha_c, sigma_c, w, p_e, theta_p, Y, p_index):
    numerator = p_index ** (1 - sigma_c) - alpha_c * p_e ** (1 - sigma_c)
    denominator = 1 - alpha_c
    p_g = (numerator / denominator) ** (1 / (1 - sigma_c))

    mc = (alpha_c * (p_e ** (1 - sigma_c)) + (1 - alpha_c) * (w ** (1 - sigma_c))) ** (
        1 / (1 - sigma_c)
    )

    N = (1 - alpha_c) * ((w / mc) ** (-sigma_c)) * Y
    E_f = alpha_c * ((p_e / mc) ** (-sigma_c)) * Y

    mu_p = theta_p / (theta_p - 1)
    profit = (1 - (mu_p ** (-1))) * Y
    return mc, E_f, N, profit, p_g


@ssj.simple
def nkpc_ss():
    inf_p = 0
    return inf_p


@ssj.simple
def nkwpc_ss(theta_w, NZ, nu_n, gamma_c, C, w):
    mu_w = theta_w / (theta_w - 1)
    v_prime = NZ**nu_n
    u_prime = C ** (-gamma_c)
    inf_w = 0
    target_w = (v_prime / u_prime) * (1 / mu_w) - w
    return inf_w, target_w


@ssj.simple
def taylor_rule_ss(r, phi_inf, inf_p):
    r_ss = r
    i = r_ss + phi_inf * inf_p
    return i, r_ss


@ssj.simple
def nkpc_dynamic(r, r_ss, phi_inf, theta_p):
    inf_p = (r - r_ss) / (phi_inf - 1)
    mu_p = theta_p / (theta_p - 1)
    return inf_p


@ssj.simple
def nkwpc_dynamic(theta_w, NZ, nu_n, gamma_c, C, inf_p, w):
    mu_w = theta_w / (theta_w - 1)
    v_prime = NZ**nu_n
    u_prime = C ** (-gamma_c)
    target_w = (v_prime / u_prime) * (1 / mu_w) - w
    inf_w = ((w - w(-1)) / w(-1)) + inf_p
    return inf_w, target_w


@ssj.simple
def taylor_rule_dynamic(r_ss, inf_p, phi_inf):
    i = r_ss + phi_inf * inf_p
    return i


@ssj.simple
def mkt_clearing(A, B, C, CE, E_f, E_s, Y, p_e, profit, r):
    asset_mkt = A - B
    energy_mkt = E_s - CE - E_f
    Q = profit + r * B
    goods_mkt = Y - p_e * E_f + Q - C
    return asset_mkt, energy_mkt, goods_mkt


def build_hank_ss_model():
    """
    Returns the steady-state HANK model object.
    """
    household_block = household_decision.add_hetinputs(
        [set_wages, make_grids, set_transfers]
    )
    household_block = household_block.add_hetoutputs(
        [get_mpc_transfers, intratemporal_consumption]
    )

    household_block = household_decision.add_hetinputs(
        [set_wages, make_grids, set_transfers]
    )
    household_block = household_block.add_hetoutputs(
        [get_mpc_transfers, intratemporal_consumption]
    )
    blocks_ss = [household_block, firm, taylor_rule_ss, nkwpc_ss, nkpc_ss, mkt_clearing]

    return ssj.create_model(blocks_ss, name="Pieroni HANK SS")
