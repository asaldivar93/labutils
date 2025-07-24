from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import xlsxwriter
from lmfit import Model


def gompertz(t, A, B, C):
    """Get gompertz model."""
    return A * np.exp(-np.exp((B * np.e / A) * (C - t) + 1))

def fit_gompertz(t, y, A, B, C, xmin):
    """Fit data to gompertz model.

    Parameters
    ----------
    t: time data
    y: growth data
    A, B, C: Initial conditions for gompertz Parameters

    """
    model = Model(gompertz)
    params = model.make_params(A=A, B=B, C=C)
    params["A"].min = xmin[0]
    params["B"].min = xmin[1]
    params["C"].min = xmin[2]
    return model.fit(y, params, t=t)

def get_growth_rates(
        biomass_df: pl.DataFrame,
        wavelengths: list[str] | None = None,
        x0: list[float] | None = None,
        xmin: list[float] | None = None,
        out_dir: str | None = None,
    ) -> dict[pl.DataFrame]:
    """Get gompertz parameters for differnt wavelengths.

    Parameters
    ----------
    file:
        an excel file with columns [wavelengths, ..., time]
    wavelengths:
        a list of wavelengths (default: [800])
    x0:
        initial guess for gompertz parameters (default: [0.1, 0.1, 20])

    """
    if wavelengths is None:
        wavelengths = ["800"]
    if x0 is None:
        x0 = [0.1, 0.1, 20]
    if xmin is None:
        xmin = [0, 0, 0]

    t = biomass_df["time"].to_numpy()
    fit_results = {}
    for wv in wavelengths:
        y = biomass_df[wv].to_numpy()
        fit_results[wv] = fit_gompertz(t, y, A=x0[0], B=x0[1], C=x0[2], xmin=xmin)

    if out_dir:
        fit_file = out_dir + "/best_fit.xlsx"
        workbook = xlsxwriter.Workbook(fit_file)
        for key, fit in fit_results.items():
            # Save report to txt
            file_path = out_dir + f"/lmfit_{key}_report"
            with Path(file_path).open("w") as file:
                file.write(fit.fit_report())

            # Save best fit to excel
            bf_df = pl.DataFrame({
                "time": t,
                "best_fit": fit.best_fit,
                "uncertainty": fit.eval_uncertainty(sigma=0.9545, t=t),
            })
            bf_df.write_excel(
                workbook,
                f"{key}",
            )
        workbook.close()

    return fit_results
