## Changes in `main`
After the merge of [#990](https://github.com/qiboteam/qibocal/pull/990), Qibocal is compatible with Qibolab 0.2.
For this reason, a small internal refactoring and some breaking changes were required.
The main differences concern the acquisition functions and the protocol parameters:
- The amplitudes are no longer relative to the values defined in the platform, but to
the maximum value the instruments reach (internally stored in `Qibolab`).
It implies renaming the amplitude parameters and converting the new amplitude level accordingly.
- The platform parameters that were previously in the `parameters.json`
but not useful for `Qibolab` to execute circuits and pulse sequences (like $E_J$, $T_1$ and $T_2$),
are moved to the `calibration.json` stored inside each platform folder.
- In the Rabi and flipping experiment, Qibocal provides the possibility to calibrate the $RX(\pi/2)$.
- Small changes in the report template.
- Some protocols are not any more supported (https://github.com/qiboteam/qibocal/pull/990#issue-2559341729
  for a list of them).
