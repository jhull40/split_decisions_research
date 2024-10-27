import pandas as pd

from constants import (
    FIGHTS_DATA_PATH,
    ODDS_DATA_PATH,
    MEAN_PROB,
    MEDIAN_PROB,
    Y_TRUE,
    Y_PRED,
)

from utils import (
    preprocess_fights,
    preprocess_odds,
    merge_fights_and_odds,
    calculate_probabilities,
    opponent_check,
    get_opponent_probabilities,
    get_agg_win_data,
    append_predictions,
)

from plots import (
    produce_histogram,
    produce_linear_win_rate_plot,
    produce_pr_curve,
    produce_roc_curve,
)


def main(column_to_use: str):
    fights = pd.read_csv(FIGHTS_DATA_PATH)
    odds = pd.read_csv(ODDS_DATA_PATH)

    fights = preprocess_fights(fights)
    odds = preprocess_odds(odds)

    df = merge_fights_and_odds(fights, odds)
    df = calculate_probabilities(df)
    df = opponent_check(df)
    df = get_opponent_probabilities(df)
    produce_histogram(df, column_to_use)
    win_data = get_agg_win_data(df, 0.05, column_to_use)
    linear_model = produce_linear_win_rate_plot(win_data, column_to_use)
    df = append_predictions(df, linear_model, column_to_use)
    produce_pr_curve(df[Y_TRUE], df[Y_PRED], column_to_use)
    produce_roc_curve(df[Y_TRUE], df[Y_PRED], column_to_use)


if __name__ == "__main__":
    column_to_use = MEAN_PROB
    main(column_to_use)
