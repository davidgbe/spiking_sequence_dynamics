from copy import deepcopy as copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats
from scipy.optimize import fsolve
import scipy.io as sio
from tqdm import tqdm
import pickle
from collections import OrderedDict
import os
from scipy.ndimage.interpolation import shift
from functools import reduce
import time
from ntwk import LIFNtwkI
from aux import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.cm as cm
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

dt = 1e-4
tau_m_e = 4e-3
tau_m_i = 4e-3
tau_a = 1.6e-3
v_th_e = 20e-3
v_th_i = 20e-3
c_e = 1e-6
c_i = 1e-6
f_e = 130
w_ee = 2.4e-4
w_ei = 0.5e-5
w_ie = -3e-5
n_exc = 250
n_inh = 50

# PARAMS
## NEURON AND NETWORK MODEL
M = Generic(
    # Excitatory membrane
    C_M_E=1e-6,  # membrane capacitance
    G_L_E=.1e-3,  # membrane leak conductance (T_M (s) = C_M (F/cm^2) / G_L (S/cm^2))
    E_L_E=-.07,  # membrane leak potential (V)
    V_TH_E=-.05,  # membrane spike threshold (V)
    T_R_E=0.5e-3,  # refractory period (s)
    T_R_I=0,
    E_R_E=-0.07, # reset voltage (V)
    
    # Inhibitory membrane
    #C_M_I=1e-6,
    #G_L_E=.1e-3, 
    #E_L_I=-.06,
    #V_TH_E=-.05,
    #T_R_I=.002,
    
    N_EXC=n_exc,
    N_INH=n_inh,
    
    # OTHER INPUTS
    SGM_N=0,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input
    
    W_E_E = w_ee,
    W_E_I = w_ei, #0.2e-5, #1e-5,
    W_I_E = w_ie / n_inh,
    W_U_E = 0,
    W_U_I = 0, #1e-1,
    
    F_IN = 1500,
    SIGMA_IN = 2e-3,
    
    F_B = 5e3,
    T_B = 15e-3,
)

t_r = M.T_R_E * np.ones((M.N_EXC + M.N_INH))
t_r[-1] = M.T_R_I


def speed_test(M, seed, buffer=0.1):
    np.random.seed(seed)
    
    w_e_i = M.W_E_I * np.random.normal(size=(M.N_INH, M.N_EXC), loc=1, scale=0.1)
    w_e_i = np.where(w_e_i > 0, w_e_i, 0)
    
    w_r = np.block([
        [ M.W_E_E * np.diag(np.ones((M.N_EXC - 1)), k=-1), M.W_I_E * np.ones((M.N_EXC, M.N_INH)) ],
        [ w_e_i, np.zeros((M.N_INH, M.N_INH)) ],
    ])

    w_u = np.block([
        [ np.array([M.W_U_E]), np.zeros((1)) ],
        [ np.zeros((M.N_EXC - 1, 2)) ],
        [ np.zeros((M.N_INH, 1)), M.W_U_I * np.ones((M.N_INH, 1)) ],
    ])

    i_b = np.zeros((M.N_EXC + M.N_INH), dtype=int)

    ntwk = LIFNtwkI(
        c_m = M.C_M_E,
        g_l = M.G_L_E,
        e_l = M.E_L_E,
        v_th = M.V_TH_E,
        v_r = M.E_R_E,
        t_r = t_r,
        w_r = w_r,
        w_u = w_u,
        i_b = i_b,
        f_b = M.F_B,
        t_b = M.T_B,
        t_a = tau_a,
        noise_var=3e-8,
    )

    S = Generic(RNG_SEED=0, T=0.62, DT=dt)
    t = np.arange(0, S.T, S.DT)

    spks_u = np.zeros((len(t), 2), dtype=int)
    
    clamp_input_spks = {}
    
    driving_pulse = np.random.poisson(lam=M.F_IN * dt, size=int(M.SIGMA_IN / dt))
    for i, val in enumerate(driving_pulse):
        if val == 1:
            clamp_input_spks[i * dt] = [0]

    rsp = ntwk.run(
        dt=S.DT,
        clamp=Generic(v={0: M.E_L_E * np.ones((M.N_EXC + M.N_INH))}, spk=clamp_input_spks),
        i_ext=np.zeros(len(t)),
        spks_u=spks_u)

    raster = np.stack([rsp.spks_t, rsp.spks_c])
    inh_raster = raster[:, raster[1, :] >= M.N_EXC]
    exc_raster = raster[:, raster[1, :] < M.N_EXC]
    
    parsed_exc_raster = exc_raster[:, exc_raster[0, :] >= buffer]
    try:
        start = parsed_exc_raster[0, parsed_exc_raster[1, :] == 50].min()
        end = parsed_exc_raster[0, parsed_exc_raster[1, :] == 100].min()
    except ValueError as e:
        return (np.nan, raster)
    print('W_E_E', M.W_E_E)
    print('W_E_I', M.W_E_I)
    return end - start, raster


def run_simulation(args):
    """Runs a single simulation instance."""
    w_ee, w_ei, seed, model_template = args
    m = copy(model_template)
    m.W_E_E = w_ee
    m.W_E_I = w_ei
    return speed_test(m, seed)


mp.set_start_method('fork', force=True)

if __name__ == '__main__':

    num_trials = 10
    seeds = np.arange(num_trials) + 3000

    w_ee_vals = np.arange(0, 0.081e-3, 0.003e-3)
    w_ei_vals = np.arange(0, 0.201e-5, 0.005e-5)

    with mp.Pool(mp.cpu_count()) as pool:
        for w_ee in w_ee_vals:
            all_p = []
            rasters = []
            for w_ei in w_ei_vals:
                # Prepare arguments for parallel execution
                args = [(w_ee, w_ei, seeds[i], M) for i in range(num_trials)]
                results = list(tqdm(pool.imap(run_simulation, args), total=num_trials))
                
                # Unpack results
                p_stable_vals, raster_vals = zip(*results)
                all_p.append(p_stable_vals)
                rasters.append(raster_vals)
                
                # Append results to file instead of overwriting
                with open('simulation_results_2.pkl', 'ab') as f:
                    pickle.dump([(w_ee, w_ei, p_stable_vals, raster_vals)], f)
