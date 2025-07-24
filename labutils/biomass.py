from __future__ import annotations

import polars as pl


def load_excel(file: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read all sheets in an excel file.

    All sheets should have the same format.

    Parameters
    ----------
    file:
        path of the data
    columns:
        list of columns to load

    Returns
    -------
    output_df:

    """
    if columns is None:
        df_dict = pl.read_excel(file, sheet_id=0)
    else:
        df_dict = pl.read_excel(file, sheet_id=0, columns=columns)
    return pl.concat(df_dict.values())

def load_tecan(
    files: list[str],
    n_wells: float | None = None,
    col_letters: str | None = None,
    keys_options: dict | None = None,
) -> pl.DataFrame:
    """Read tecan excel file with multiple sheets.

    Parameters
    ----------
    files:
        A list of file paths to tecan files
    n_wells:
        The number of wells in each sheet. It will be tha same for all sheets.
        (default: 4)
    col_letters:
        The columns to read. (default: "A:AE")
    keys_options:
        (Optional) Keys mapping wells to sample id. A dict with keys skip_rows,
        n_rows, use_columns

    """
    if n_wells is None:
        n_wells = 4
    if col_letters is None:
        col_letters = "A:AE"

    files_dict = {}
    for f in files:
        # read Absorbance values
        start_row = 31
        read_options = {
            "header_row": start_row,
            "n_rows": n_wells,
            "use_columns": col_letters,
        }
        df_dict = pl.read_excel(
            f,
            sheet_id=0,
            read_options=read_options,
            has_header=True,
        )
        for k, v in df_dict.items():
            df_dict[k] = v.rename({"Wavel.": "well"})

        # Add well to sample key if available
        if keys_options:
            keys_dict = pl.read_excel(f, sheet_id=0, read_options=keys_options)
            for sheet, v in keys_dict.items():
                df_dict[sheet] = df_dict[sheet].join(v, on="well")

        # Get the date for each measurment
        start_row = 27
        read_options = {
            "skip_rows": start_row,
            "n_rows": 1,
            "use_columns": "B",
        }
        dates = pl.read_excel(f, sheet_id=0, read_options=read_options)
        # Add date to absorbance values
        for sheet, v in dates.items():
            v = v.with_columns(
                pl.col("__UNNAMED__1").str.to_datetime("%m/%d/%Y %I:%M:%S %p"),
            )
            df_dict[sheet] = df_dict[sheet].with_columns(date = v[0, 0])

        files_dict[f] = pl.concat(df_dict.values())
    od_df = pl.concat(files_dict.values()).sort("date")
    od_df = od_df.with_columns(time = pl.col("date") - od_df[0, -1])
    return od_df.with_columns(time = pl.col("time").dt.total_seconds() / (3600 * 24))

def get_od_df(file: str, wavelengths: list[str] | None) -> pl.DataFrame:
    """Get a dataframe of selected wavelengths."""
    if wavelengths is None:
        wavelengths = ["440", "680", "800"]

    od_df = load_excel(file, wavelengths)
    od_df = od_df.sort("date")
    return od_df.with_columns(time = pl.col("date") - od_df[0, 3]).sort("time")

def get_dw_df(file: str) -> pl.DataFrame:
    """Get dry weight dataframe."""
    columns = ["type", "Biomass (g/L)", "date"]
    dw_df = load_excel(file, columns)

    # Keep only volatile suspended solids
    # get mean and std of concentration
    # sort by date (oldest to newest)
    dw_df = dw_df.filter(
        pl.col("type") == "VSS",
    )
    dw_df = dw_df.group_by("date").agg(
        pl.mean("Biomass (g/L)"),
        std=pl.std("Biomass (g/L)"),
    ).sort("date")

    return dw_df.with_columns(time = pl.col("date") - dw_df[0, 0])
