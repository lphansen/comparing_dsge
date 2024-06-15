import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator as RGI
from tqdm import tqdm

def simulate_single_process(n, initial_point, statespace, dt, f_muX, f_sigmaX, f_mulogM, f_sigmalogM, f_mulogS, f_sigmalogS, sim_length):
    successful_results = []
    success_count = 0
    for series in range(100):
        try:
            np.random.seed(n * 100 + series)  
            n_X = len(statespace)
            n_W = len(f_sigmaX[0])
            T = sim_length

            W_series = np.random.multivariate_normal(np.zeros(n_W), np.eye(n_W) * dt, T)
            X_process = np.zeros([n_X, T])
            dlogM_process = np.zeros(T)
            dlogS_process = np.zeros(T)

            for i in range(n_X):
                X_process[i, 0] = initial_point[i]
            for t in range(T - 1):
                muX_t = [f_mux([X_process[i, t] for i in range(n_X)]) for f_mux in f_muX]
                sigmaX_t = [[sigma([X_process[i, t] for i in range(n_X)]) for sigma in sigmaX] for sigmaX in f_sigmaX]
                mulogM_t = f_mulogM([X_process[i, t] for i in range(n_X)])
                sigmalogM_t = [sigma([X_process[i, t] for i in range(n_X)]) for sigma in f_sigmalogM]
                mulogS_t = f_mulogS([X_process[i, t] for i in range(n_X)])
                sigmalogS_t = [sigma([X_process[i, t] for i in range(n_X)]) for sigma in f_sigmalogS]
                for i in range(n_X):
                    X_process[i, t + 1] = X_process[i, t] + muX_t[i] * dt + np.sum([W_series[t, s] * sigmaX_t[i][s] for s in range(n_W)])
                dlogM_process[t + 1] = mulogM_t * dt + np.sum([W_series[t, s] * sigmalogM_t[s] for s in range(n_W)])
                dlogS_process[t + 1] = mulogS_t * dt + np.sum([W_series[t, s] * sigmalogS_t[s] for s in range(n_W)])
            MtM0_process = np.exp(np.cumsum(dlogM_process))
            SMtSM0_process = np.exp(np.cumsum(dlogS_process)) * np.exp(np.cumsum(dlogM_process))

            successful_results.append((X_process, dlogM_process, dlogS_process, MtM0_process, SMtSM0_process))
            success_count += 1
        except:
            continue  

    return successful_results, success_count

def simulate_elasticities(statespace, shock_index, dx, dt, muX, sigmaX, mulogM, sigmalogM, mulogS, sigmalogS, initial_point, sim_length, sim_num, return_type=1):
    """
    Computes the shock elasticity of the model using the simulation method.

    Parameters:
    - statespace: list of flatten numpy arrays
        List of state variables in flattened numpy arrays for each state dimension. Grid points should be unique and sorted in ascending order.
    - shock_index: int
        The index of the shock to compute the elasticity for.
    - dx: list of floats or ints
        List of small changes to the initial state variables for the derivative in shock elasticity.
    - dt: float or int
        Time step for the shock elasticity PDE.
    - muX: list of numpy arrays
        List of state variable drift terms as numpy arrays, evaluated at the grid points. The order should match the state variables.
    - sigmaX: list of lists of numpy arrays
        List of lists of state variable diffusion terms as numpy arrays, corresponding to different shocks. Evaluated at the grid points. The order should match the state variables.
    - mulogM: numpy array
        The log drift term for the response variable M.
    - sigmalogM: list of numpy arrays
        The log diffusion terms for the response variable M.
    - mulogS: numpy array
        The log drift term for the SDF.
    - sigmalogS: list of numpy arrays
        The log diffusion terms for the SDF.
    - initial_point: list of floats or ints
        List of a single initial state variable point for the shock elasticity.
    - T: float
        The calculation period for the shock elasticity given dt.
    - sim_length: int
        The length of the simulation for the process.
    - sim_num: int
        The number of simulations to run divided by 100. The actual number of simulations will be 100 times this number.
    - return_type: int
        The return type for the function. 1 for only the elasticities, 2 for the elasticities and the simulation processes.

    Returns:
    dict
        A dictionary with the computed exposure and price elasticities, and the success counts for the initial and derivative points.
    """
    derivative_points = []
    for i in range(len(initial_point)):
        derivative_point = initial_point[:]
        derivative_point[i] += dx[i]
        derivative_points.append(derivative_point)

    f_muX = [RGI(statespace, mux, fill_value=None, bounds_error=True) for mux in muX]
    f_sigmaX = [[RGI(statespace, sigma, fill_value=None, bounds_error=True) for sigma in sigmax] for sigmax in sigmaX]
    f_mulogM = RGI(statespace, mulogM, fill_value=None, bounds_error=True)
    f_sigmalogM = [RGI(statespace, sigma, fill_value=None, bounds_error=True) for sigma in sigmalogM]
    f_mulogS = RGI(statespace, mulogS, fill_value=None, bounds_error=True)
    f_sigmalogS = [RGI(statespace, sigma, fill_value=None, bounds_error=True) for sigma in sigmalogS]

    def run_simulations(initial_point):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(simulate_single_process, n, initial_point, statespace, dt, f_muX, f_sigmaX, f_mulogM, f_sigmalogM, f_mulogS, f_sigmalogS, sim_length) for n in tqdm(range(sim_num), desc="Submitting tasks")]
            results = []
            success_count = 0
            for future in tqdm(as_completed(futures), total=sim_num, desc="Processing tasks"):
                result, count = future.result()
                results.extend(result)
                success_count += count

        if not results:
            return None, None, None, None, None, None, None, success_count

        X_processes, dlogM_processes, dlogS_processes, MtM0_processes, SMtSM0_processes = zip(*results)
        log_conditional_expectation_MtM0 = np.log(np.mean(np.array(MtM0_processes), axis=0))
        log_conditional_expectation_SMtSM0 = np.log(np.mean(np.array(SMtSM0_processes), axis=0))
        return log_conditional_expectation_MtM0, log_conditional_expectation_SMtSM0, X_processes, dlogM_processes, dlogS_processes, MtM0_processes, SMtSM0_processes, success_count

    log_conditional_expectation_MtM0, log_conditional_expectation_SMtSM0, X_processes, dlogM_processes, dlogS_processes, MtM0_processes, SMtSM0_processes, initial_success_count = run_simulations(initial_point)
    
    if log_conditional_expectation_MtM0 is None or log_conditional_expectation_SMtSM0 is None:
        raise RuntimeError("All simulations failed for the initial point.")

    log_conditional_expectation_MtM0_derivative = []
    log_conditional_expectation_SMtSM0_derivative = []
    X_processes_derivative = []
    dlogM_processes_derivative = []
    dlogS_processes_derivative = []
    MtM0_processes_derivative = []
    SMtSM0_processes_derivative = []
    derivative_success_counts = []

    for derivative_point in derivative_points:
        log_conditional_expectation_MtM0_temp, log_conditional_expectation_SMtSM0_temp, X_processes_temp, dlogM_processes_temp, dlogS_processes_temp, MtM0_processes_temp, SMtSM0_processes_temp, derivative_success_count = run_simulations(derivative_point)
        derivative_success_counts.append(derivative_success_count) 
        if log_conditional_expectation_MtM0_temp is None or log_conditional_expectation_SMtSM0_temp is None:
            continue
        log_conditional_expectation_MtM0_derivative.append(log_conditional_expectation_MtM0_temp)
        log_conditional_expectation_SMtSM0_derivative.append(log_conditional_expectation_SMtSM0_temp)
        X_processes_derivative.append(X_processes_temp)
        dlogM_processes_derivative.append(dlogM_processes_temp)
        dlogS_processes_derivative.append(dlogS_processes_temp)
        MtM0_processes_derivative.append(MtM0_processes_temp)
        SMtSM0_processes_derivative.append(SMtSM0_processes_temp)

    exposure_derivative = [(i - log_conditional_expectation_MtM0) / dx[index] for index, i in enumerate(log_conditional_expectation_MtM0_derivative)]
    price_derivative = [(i - log_conditional_expectation_SMtSM0) / dx[index] for index, i in enumerate(log_conditional_expectation_SMtSM0_derivative)]

    sigmaX0 = [[sigma(initial_point) for sigma in sigmaX] for sigmaX in f_sigmaX]
    sigmalogM0 = [sigma(initial_point) for sigma in f_sigmalogM]
    sigmalogS0 = [sigma(initial_point) for sigma in f_sigmalogS]

    expo_list = [sigmaX0[i][shock_index] * exposure_derivative[i] for i in range(len(initial_point))]
    exposure_elasticity = np.array([sum(values) for values in zip(*expo_list)]) + sigmalogM0[shock_index]
    price_list = [sigmaX0[i][shock_index] * price_derivative[i] for i in range(len(initial_point))]
    price_elasticity = exposure_elasticity - (np.array([sum(values) for values in zip(*price_list)]) + sigmalogM0[shock_index] + sigmalogS0[shock_index])

    if return_type == 1:
        return {
            'exposure_elasticity': exposure_elasticity,
            'price_elasticity': price_elasticity,
            'initial_success_count': initial_success_count,
            'derivative_success_counts': derivative_success_counts
        }
    else:
        return {
            'exposure_elasticity': exposure_elasticity,
            'price_elasticity': price_elasticity,
            'initial_success_count': initial_success_count,
            'derivative_success_counts': derivative_success_counts
        }, {
            'X_processes': X_processes,
            'dlogM_processes': dlogM_processes,
            'dlogS_processes': dlogS_processes,
            'MtM0_processes': MtM0_processes,
            'SMtSM0_processes': SMtSM0_processes,
            'X_processes_derivative': X_processes_derivative,
            'dlogM_processes_derivative': dlogM_processes_derivative,
            'dlogS_processes_derivative': dlogS_processes_derivative,
            'MtM0_processes_derivative': MtM0_processes_derivative,
            'SMtSM0_processes_derivative': SMtSM0_processes_derivative
        }
