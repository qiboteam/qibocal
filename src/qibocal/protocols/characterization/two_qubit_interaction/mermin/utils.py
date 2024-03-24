def compute_mermin(frequencies):
    """Computes the chsh inequality out of the frequencies of the 4 circuits executed."""
    m = 0
    aux = 0
    for freq in frequencies:
        for key in freq.keys():
            if (
                aux == 3
            ):  # This value sets where the minus sign is in the CHSH inequality
                m -= freq[key] * (-1) ** (sum([int(key[i]) for i in range(len(key))]))
            else:
                m += freq[key] * (-1) ** (sum([int(key[i]) for i in range(len(key))]))
        aux += 1
    nshots = sum(freq[x] for x in freq)
    return m / nshots
