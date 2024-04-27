def qubit_spectroscopy_flux_fit(data, x, y, qubits, resonator_type, transition):
    r""" """
    data_fit = Data(
        name=f"fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "qubit",
            "fluxline",
        ],
    )
    bias_keys = parse(x)
    frequency_keys = parse(y)

    for qubit in qubits:
        fluxlines = data.df[data.df["qubit"] == qubit]["fluxline"].unique()
        for fluxline in fluxlines:
            qubit_data = (
                data.df[(data.df["qubit"] == qubit) & (data.df["fluxline"] == fluxline)]
                .drop(columns=["i", "q", "phase", "qubit", "fluxline", "iteration"])
                .groupby([bias_keys[0], frequency_keys[0]], as_index=False)
                .mean()
            )
            biases = (
                qubit_data[bias_keys[0]].pint.to(bias_keys[1]).pint.magnitude.unique()
            )
            if resonator_type == "3D":
                frequencies = (
                    qubit_data.loc[qubit_data.groupby(bias_keys[0])["MSR"].idxmin()][
                        "frequency"
                    ]
                    .pint.to(frequency_keys[1])
                    .pint.magnitude.to_numpy()
                )
            else:
                frequencies = (
                    qubit_data.loc[qubit_data.groupby(bias_keys[0])["MSR"].idxmax()][
                        "frequency"
                    ]
                    .pint.to(frequency_keys[1])
                    .pint.magnitude.to_numpy()
                )
            try:
                if qubit == fluxline:
                    popt = np.polyfit(biases, frequencies, 2)
                else:
                    popt_line = np.polyfit(biases, frequencies, 1)
                    popt = [0, popt_line[0], popt_line[1]]
            except:
                log.warning(
                    "qubit_spectroscopy_flux_fit: the fitting was not succesful"
                )
                data_fit.add({key: 0 for key in data_fit.df.columns})
                return data_fit
            data_fit.add(
                {
                    "popt0": popt[0],
                    "popt1": popt[1],
                    "popt2": popt[2],
                    "qubit": qubit,
                    "fluxline": fluxline,
                }
            )
    return data_fit
