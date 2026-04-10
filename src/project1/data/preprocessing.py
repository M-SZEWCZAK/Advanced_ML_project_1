"""Preprocessing utilities for Project 1."""


def add_intercept_column(df):
    """Insert an intercept column at the beginning of a DataFrame."""
    df.insert(0, "intercept", 1)
    return df


__all__ = ["add_intercept_column"]
