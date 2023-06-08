import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def ramsey_fit_and_analyze(
    x_data,
    y_data,
    initial_guess_function,
    model_function,
    n_bootstrap,
    max_t1,
    r_squared_limit=0.78,
):
    """
    Perform Ramsey fit and analysis on given data.

    Args:
        x_data (np.ndarray): x-axis data.
        y_data (np.ndarray): y-axis data.
        initial_guess_function (callable): Function to compute initial guesses for curve fitting.
        model_function (callable): Function representing the model to fit.
        n_bootstrap (int): Number of bootstrap iterations.
        max_t1 (float): Maximum T1 value.
        r_squared_limit (float, optional): Minimum R-squared value to consider as a good fit. Default is 0.78.

    Returns:
        dict: Dictionary containing fit results and analysis.

    Example:
        .. testcode::

            import numpy as np
            from qibocal.fitting.methods import ramsey
            from qibocal.fitting.noise_models.initial_guess import ramsey_initial_guess_direct_interpolation
            from qibocal.fitting.noise_models.bootstrapping_for_ramsey import ramsey_fit_and_analyze

            x_data = np.array([0, 1, 2, 3, 4, 5])
            y_data = np.array([0.2, 0.5, 0.7, 0.9, 1.0, 0.8])

            result = ramsey_fit_and_analyze(x_data, y_data,
                                            initial_guess_function=ramsey_initial_guess_direct_interpolation,
                                            model_function=ramsey,
                                            n_bootstrap=1000,
                                            max_t1=100000,
                                            r_squared_limit=0.78)
    """

    # Normalize x_data and y_data to the range [0, 1]
    x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
    y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

    # Calculate the mean value for each step on the x-axis
    unique_x = np.unique(x_data)
    y_mean = [np.mean(y_data[x_data == x]) for x in unique_x]

    # Define the parameter bounds for the curve_fit function
    bounds = (
        [0, np.min(y_mean), 0, -np.pi, 0],
        [np.inf, np.max(y_mean), np.inf, np.pi, 2 * max_t1],
    )

    # Perform the initial curve fitting using the provided data
    initial_guess = initial_guess_function(unique_x, y_mean)
    popt, _ = curve_fit(
        model_function,
        unique_x,
        y_mean,
        p0=initial_guess,
        maxfev=4000000,
        bounds=bounds,
    )

    # Perform non-parametric bootstrap for n_bootstrap iterations
    bootstrap_params = []
    bootstrap_y_estimates = []
    for i in range(n_bootstrap):
        bootstrap_y = []
        for x in unique_x:
            # Perform non-parametric bootstrap by randomly sampling y_data with replacement
            bootstrap_y_data = np.random.choice(
                y_data[x_data == x], size=len(y_data[x_data == x]), replace=True
            )
            bootstrap_y.append(np.mean(bootstrap_y_data))
        bootstrap_y_estimates.append(bootstrap_y)

        # Fit the curve to the bootstrap sample
        initial_guess = initial_guess_function(unique_x, bootstrap_y)
        bootstrap_popt, _ = curve_fit(
            model_function,
            unique_x,
            bootstrap_y,
            p0=initial_guess,
            maxfev=400000,
            bounds=bounds,
        )
        # Calculate the residuals
        residuals = bootstrap_y - model_function(unique_x, *bootstrap_popt)

        # Calculate the total sum of squares (TSS)
        tss = np.sum((bootstrap_y - np.mean(bootstrap_y)) ** 2)

        # Calculate the residual sum of squares (RSS)
        rss = np.sum(residuals**2)

        # Calculate the R-squared value
        r_squared = 1 - (rss / tss)

        # Clip the parameter values to the defined bounds
        bootstrap_popt = np.clip(bootstrap_popt, bounds[0], bounds[1])

        if r_squared > r_squared_limit:
            bootstrap_params.append(bootstrap_popt)

    # Convert the bootstrap parameter estimates to a numpy array
    bootstrap_params = np.array(bootstrap_params)

    # Compute the parameter errors as the standard deviation of the bootstrap parameter estimates
    perrs = np.array(3 * np.std(bootstrap_params, axis=0))
    errors = 3 * np.std(bootstrap_y_estimates, axis=0)
    dof = len(unique_x) - len(popt) - 1
    chi2 = np.sum((model_function(unique_x, *popt) - y_mean) ** 2 / errors**2)
    chi2_dist = stats.chi2(dof)
    reduced_chi2 = chi2 / dof
    p_value = 1 - chi2_dist.cdf(chi2)

    # Return the fit results as a dictionary
    return {
        "popt": popt,
        "perr": perrs,
        "reduced_chi2": reduced_chi2,
        "p_value": p_value,
        "y_mean": y_mean,
        "bootstrap_y_estimates": bootstrap_y_estimates,
        "errors": errors,
    }


def calculate_max_t1(w_r, w_q, g, k):
    """
    Calculate the maximum T1 using the Purcell effect.

    The Purcell effect describes the modification of the spontaneous emission rate
    of a quantum system due to its interaction with a resonant electromagnetic mode.
    This function calculates the maximum T1 time, which is the time it takes for the
    qubit to decay, by considering the difference between the resonator and qubit frequencies,
    the coupling strength, and the photon loss rate from the resonator.

    The frequencies, coupling strength, and loss rate are initially provided in Hz.

    The formula used to calculate the maximum T1 is:

    ..math::
        T_1^{(\\max)} = (w_r - w_q)^2 / (g^2 \\cdot k)

    ..note::
        Ensure that the input values for the frequencies, coupling strength, and
        photon loss rate are appropriate for the specific quantum system being considered.

    Args:
        w_r: Resonator frequency in Hz.
        w_q: Qubit frequency in Hz.
        g: Coupling strength in Hz.
        k: Photon loss rate from the resonator in Hz.

    Returns:
        max_t1: Maximum T1 time.


    Example:
        .. testcode::

            from qibocal.fitting.noise_models.bootstrapping_for_ramsey import calculate_max_t1

            w_r = 7.8e9
            w_q = 5.5e9
            g = 100e6
            k = 1e6
            max_t1 = calculate_max_t1(w_r, w_q, g, k)
            print(max_t1)
    """

    max_t1 = (w_r - w_q) ** 2 / (g**2 * k)
    return max_t1
