"""
# New Protocol Interface Demo

This notebook demonstrates the new magic-free protocol execution interface.
"""

from dataclasses import dataclass
from typing import Optional

import marimo as mo
import numpy as np

__generated_with = "0.10.18"

app = mo.App()


@app.cell
def __():
    import sys
    from pathlib import Path

    # Add qibocal to path for imports
    qibocal_path = Path.cwd().parent / "src"
    if str(qibocal_path) not in sys.path:
        sys.path.insert(0, str(qibocal_path))

    return qibocal_path


@app.cell
def __():
    mo.md("""
    ## New Protocol Interface Demo

    This notebook demonstrates the new magic-free protocol execution interface introduced in #1456.

    Key improvements:
    - **No magic**: Explicit control over execution flow
    - **Type-safe**: Full type hints for all protocol phases
    - **Flexible**: Support for optional phases (fit, report, update)
    - **Composable**: Easy to build complex workflows
    """)


@app.cell
def __():
    import numpy as np

    from qibocal.auto.execute import Executor
    from qibocal.auto.operation import BoundProtocol, Completed, Protocol

    return Protocol, BoundProtocol, Completed, Executor, dataclass, Optional, np


@app.cell
def __(dataclass):
    mo.md("## Step 1: Define Parameters")


@app.cell
def __(dataclass):
    @dataclass
    class SpectroscopyParams:
        """Parameters for a simple spectroscopy experiment."""

        start_freq: float
        stop_freq: float
        n_points: int
        amplitude: float = 0.1

    return SpectroscopyParams


@app.cell
def __():
    mo.md("## Step 2: Define Data Structure")


@app.cell
def __(dataclass):
    @dataclass
    class SpectroscopyData:
        """Data collected from spectroscopy experiment."""

        frequencies: np.ndarray
        signal: np.ndarray
        metadata: dict

    return SpectroscopyData


@app.cell
def __():
    mo.md("## Step 3: Define Results Structure")


@app.cell
def __(dataclass):
    @dataclass
    class SpectroscopyResults:
        """Fitted results from spectroscopy."""

        resonance_freq: float
        linewidth: float
        quality_factor: float
        fit_params: dict

    return SpectroscopyResults


@app.cell
def __():
    mo.md("## Step 4: Implement Protocol Functions")


@app.cell
def __(SpectroscopyParams, SpectroscopyData, np):
    def acquire_spectroscopy(params: SpectroscopyParams) -> SpectroscopyData:
        """Simulate acquiring spectroscopy data."""
        # Simulate frequency response
        freqs = np.linspace(params.start_freq, params.stop_freq, params.n_points)
        resonance = (params.start_freq + params.stop_freq) / 2
        linewidth = (params.stop_freq - params.start_freq) / 10

        # Lorentzian response
        signal = params.amplitude / (1 + ((freqs - resonance) / linewidth) ** 2)
        # Add noise
        signal += np.random.normal(0, 0.01, len(signal))

        return SpectroscopyData(
            frequencies=freqs,
            signal=signal,
            metadata={"n_shots": 1000, "readout_delay": 100},
        )

    return acquire_spectroscopy


@app.cell
def __(SpectroscopyData, SpectroscopyResults, np, Optional):
    def fit_spectroscopy(
        data: SpectroscopyData, fit_params: Optional[dict] = None
    ) -> SpectroscopyResults:
        """Fit spectroscopy data to find resonance."""
        # Find peak
        peak_idx = np.argmax(data.signal)
        resonance_freq = data.frequencies[peak_idx]

        # Estimate linewidth
        half_max = data.signal[peak_idx] / 2
        above_half = np.where(data.signal > half_max)[0]
        if len(above_half) > 1:
            linewidth = (
                data.frequencies[above_half[-1]] - data.frequencies[above_half[0]]
            )
        else:
            linewidth = (data.frequencies[-1] - data.frequencies[0]) / 10

        # Calculate quality factor
        q_factor = resonance_freq / linewidth if linewidth > 0 else 0

        return SpectroscopyResults(
            resonance_freq=resonance_freq,
            linewidth=linewidth,
            quality_factor=q_factor,
            fit_params={"method": "peak_finding", "version": 1},
        )

    return fit_spectroscopy


@app.cell
def __():
    mo.md("## Step 5: Reporting Function (Optional)")


@app.cell
def __(SpectroscopyData, SpectroscopyResults, mo, Optional):
    def report_spectroscopy(
        data: SpectroscopyData,
        results: SpectroscopyResults,
        report_params: Optional[dict] = None,
    ) -> None:
        """Generate report for spectroscopy results."""
        report_text = f"""
        ### Spectroscopy Results

        **Resonance Frequency**: {results.resonance_freq:.6e} Hz
        **Linewidth**: {results.linewidth:.6e} Hz
        **Quality Factor**: {results.quality_factor:.2f}

        **Metadata**: {data.metadata}
        """
        mo.md(report_text)

    return report_spectroscopy


@app.cell
def __():
    mo.md("## Step 6: Create Protocol Instance")


@app.cell
def __(Protocol, acquire_spectroscopy, fit_spectroscopy, report_spectroscopy):
    # Create protocol with all phases
    spectroscopy_protocol = Protocol(
        acquisition=acquire_spectroscopy,
        fit=fit_spectroscopy,
        report=report_spectroscopy,
    )

    return spectroscopy_protocol, mo.md(
        "✓ Protocol created with phases: acquisition, fit, report"
    )


@app.cell
def __():
    mo.md("## Step 7: Bind Parameters to Protocol")


@app.cell
def __(spectroscopy_protocol, SpectroscopyParams, mo):
    # Create parameters
    params = SpectroscopyParams(
        start_freq=5e9, stop_freq=5.5e9, n_points=50, amplitude=0.5
    )

    # Bind to protocol
    bound = spectroscopy_protocol(pars=params)

    return (
        bound,
        params,
        mo.md(f"""
    ✓ Bound protocol with:
    - Start frequency: {params.start_freq / 1e9:.1f} GHz
    - Stop frequency: {params.stop_freq / 1e9:.1f} GHz
    - Number of points: {params.n_points}
    """),
    )


@app.cell
def __():
    mo.md("## Step 8: Execute Protocol (Mock Platform)")


@app.cell
def __(Executor):
    # Create mock platform (would be real hardware in production)
    class MockPlatform:
        """Mock platform for testing."""

        def update(self, results):
            pass

    executor = Executor(MockPlatform())
    return executor, MockPlatform


@app.cell
def __(executor, bound, mo):
    # Execute complete workflow
    result = executor(bound)

    return result, mo.md(f"""
    ✓ Execution completed successfully!

    - Acquired data: {result.data.signal.shape[0]} points
    - Resonance frequency: {result.results.resonance_freq / 1e9:.4f} GHz
    - Linewidth: {result.results.linewidth / 1e9:.4f} GHz
    - Quality factor: {result.results.quality_factor:.1f}
    """)


@app.cell
def __():
    mo.md("## Step 9: Partial Execution (Acquire Only)")


@app.cell
def __(executor, bound, mo):
    # We can also execute phases separately
    # For example, skip fitting and just acquire
    data = executor.acquire(bound)

    return data, mo.md(f"""
    ✓ Acquired data independently:
    - Signal shape: {data.signal.shape}
    - Metadata: {data.metadata}
    """)


@app.cell
def __(executor, data, bound, mo):
    # Now fit the acquired data
    results = executor.fit(data, bound)

    return results, mo.md(f"""
    ✓ Fitted pre-acquired data:
    - Resonance: {results.resonance_freq / 1e9:.4f} GHz
    - Q-factor: {results.quality_factor:.1f}
    """)


@app.cell
def __():
    mo.md("""
    ## Summary

    The new protocol interface provides:

    1. **Protocol Definition**: Type-safe definition with acquisition, fit, report, update
    2. **Parameter Binding**: Explicit binding of parameters to protocols
    3. **Flexible Execution**: Execute complete workflows or individual phases
    4. **No Magic**: Full control over execution flow
    5. **Composability**: Easy to build complex workflows

    ### Benefits over Legacy Interface

    - ✓ No implicit magic in protocol execution
    - ✓ Full type hints for all phases
    - ✓ Optional phases (fit, report, update)
    - ✓ Can execute phases independently
    - ✓ Simpler mental model
    - ✓ Easier to test and debug
    """)


if __name__ == "__main__":
    app.run()
