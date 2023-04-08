import numpy as np


def clean(series):
    series = series.str.upper()
    series = series.str.replace("Á", "A")
    series = series.str.replace("É", "E")
    series = series.str.replace("Ú", "U")
    series = series.str.replace("Ó", "O")
    series = series.str.replace("Í", "I")
    series = series.str.replace("Ã", "A")
    series = series.str.replace("Õ", "O")
    series = series.str.replace("Ô", "O")
    series = series.str.replace("Â", "A")
    series = series.str.replace("Ê", "E")
    series = series.str.replace("Ç", "C")
    series = series.str.replace("-", "")
    series = series.str.replace("'", "")
    series = series.str.upper()
    return series


def clean_str(series):
    series = series.upper()
    series = series.replace("Á", "A")
    series = series.replace("É", "E")
    series = series.replace("Ú", "U")
    series = series.replace("Ó", "O")
    series = series.replace("Í", "I")
    series = series.replace("Ã", "A")
    series = series.replace("Õ", "O")
    series = series.replace("Ô", "O")
    series = series.replace("Â", "A")
    series = series.replace("Ê", "E")
    series = series.replace("Ç", "C")
    series = series.replace("-", "")
    series = series.replace("'", "")
    return series


def group_data(df, list):
    grouped_df = (
        df.groupby(list)
        .agg(
            {
                "last_available_confirmed": "sum",
                "last_available_confirmed_per_100k_inhabitants": "mean",
                "new_confirmed": "sum",
                "last_available_deaths": "sum",
                "new_deaths": "sum",
                "estimated_population": "mean",
            }
        )
        .reset_index()
    )
    return grouped_df


def create_variable(df, level, down_limit, up_limit):
    df["confirmed_percent"] = df.last_available_confirmed / df.estimated_population
    df["new_confirmed_percent_pop"] = df.new_confirmed / df.estimated_population
    df["new_deaths_percent"] = df.groupby(level)["new_deaths"].pct_change()
    df["new_confirmed_percent"] = df.groupby(level)["new_confirmed"].pct_change()
    df["new_deaths_percent"] = df["new_deaths_percent"].replace(
        [np.inf, -np.inf, np.nan, None], None
    )
    df["new_confirmed_percent"] = df["new_confirmed_percent"].replace(
        [np.inf, -np.inf, np.nan, None], None
    )
    df["decrease_deaths"] = df["new_deaths_percent"] <= down_limit
    df["stability_deaths"] = (df["new_deaths_percent"] > down_limit) & (
        df["new_deaths_percent"] <= up_limit
    )
    df["increase_deaths"] = df["new_deaths_percent"] > up_limit
    df["decrease_cases"] = df["new_confirmed_percent"] < down_limit
    df["stability_cases"] = (df["new_confirmed_percent"] > down_limit) & (
        df["new_confirmed_percent"] <= up_limit
    )
    df["increase_cases"] = df["new_confirmed_percent"] > up_limit
    df["last_week_confirmed"] = [None] + list(df["last_available_confirmed"][:-1])
    df["last_week_new_confirmed"] = [None] + list(df["new_confirmed"][:-1])
    return df


def group_time(df, level_time, level_city):
    grouped_df = df.groupby(level_time).agg(
        {
            level_city: "count",
            "decrease_deaths": "sum",
            "stability_deaths": "sum",
            "increase_deaths": "sum",
            "decrease_cases": "sum",
            "stability_cases": "sum",
            "increase_cases": "sum",
        }
    )
    grouped_df["decrease_deaths_percent"] = (
        grouped_df["decrease_deaths"] / grouped_df[level_city]
    )
    grouped_df["stability_deaths_percent"] = (
        grouped_df["stability_deaths"] / grouped_df[level_city]
    )
    grouped_df["increase_deaths_percent"] = (
        grouped_df["increase_deaths"] / grouped_df[level_city]
    )
    grouped_df["decrease_cases_percent"] = (
        grouped_df["decrease_cases"] / grouped_df[level_city]
    )
    grouped_df["stability_cases_percent"] = (
        grouped_df["stability_cases"] / grouped_df[level_city]
    )
    grouped_df["increase_cases_percent"] = (
        grouped_df["increase_cases"] / grouped_df[level_city]
    )
    return grouped_df
