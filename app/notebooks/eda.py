import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Census data: Basic Exploratory Data Analysis
    Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income)

    Author: Diego Hernández Jiménez
    """)


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell
def _(pd):
    df = pd.read_csv("./data/census.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    df.info()


@app.cell
def _(df):
    df.columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove extra spaces
    """)


@app.cell
def _(df):
    # column extra spaces
    df.columns = [col.strip() for col in df.columns]

    # row values extra spaces
    str_cols = df.select_dtypes(include="object").columns

    df[str_cols] = df[str_cols].apply(lambda val: val.str.strip())
    return (str_cols,)


@app.cell
def _(df, str_cols):
    def categorical_unique_values(df, str_cols):
        for str_col in str_cols:
            print(f"Unique values '{str_col}'")
            print(df[str_col].unique(), "\n")

    categorical_unique_values(df, str_cols)
    return (categorical_unique_values,)


@app.cell
def _(categorical_unique_values, df, np, str_cols):
    df[str_cols] = df[str_cols].replace("?", np.nan)
    categorical_unique_values(df, str_cols)


@app.cell
def _(df):
    df.select_dtypes(int).describe()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions and next steps

    - Preprocess step: remove spaces from column names
    - Preprocess step: remove spaces from row values (categorical columns)
    - Preprocess step: consider "?" as null
    """)


if __name__ == "__main__":
    app.run()
