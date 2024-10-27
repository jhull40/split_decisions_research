import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from constants import (
    Y_PRED,
    Y_TRUE,
    BET_COLS,
    MEAN_PROB,
    MEDIAN_PROB,
)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])

    df['red_fighter'] = df['red_fighter'].str.title()
    df['blue_fighter'] = df['blue_fighter'].str.title()

    return df


def preprocess_fights(fights: pd.DataFrame) -> pd.DataFrame:

    fights['red_fighter'] = (
        fights['red_fighter__first_name'] + ' ' + fights['red_fighter__last_name']
    )
    fights['blue_fighter'] = (
        fights['blue_fighter__first_name'] + ' ' + fights['blue_fighter__last_name']
    )
    fights = fights.drop(
        [
            'Unnamed: 0',
            'red_fighter__first_name',
            'red_fighter__last_name',
            'blue_fighter__first_name',
            'blue_fighter__last_name',
        ],
        axis=1,
    )

    fights = standardize_columns(fights)

    return fights


def preprocess_odds(odds: pd.DataFrame) -> pd.DataFrame:
    odds = odds[odds['bet_type'] == 'Ordinary']
    odds = odds.rename(
        columns={
            'Card_Date': 'date',
            'fighter1': 'blue_fighter',
            'fighter2': 'red_fighter',
        }
    )

    odds = standardize_columns(odds)

    return odds


def merge_fights_and_odds(fights: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged1 = odds.merge(fights, how='left', on=['red_fighter', 'blue_fighter', 'date'])
    matched1 = merged1[merged1['id'].notna()]
    to_flip = (
        merged1[merged1['id'].isna()]
        .drop(columns=['id', 'winner', 'ufc_stats_id'], axis=1)
        .copy()
    )

    to_flip = to_flip.rename(
        columns={'blue_fighter': 'red_fighter', 'red_fighter': 'blue_fighter'}
    )

    merged2 = to_flip.merge(
        fights, how='left', on=['red_fighter', 'blue_fighter', 'date']
    )
    matched2 = merged2[merged2['id'].notna()]

    df = pd.concat([matched1, matched2], ignore_index=True)

    df['fighter'] = np.where(df['Bet'] == df['blue_fighter'], 'blue', 'red')

    df[Y_TRUE] = np.where(df['fighter'] == df['winner'], 1, 0)

    return df


def calculate_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    for col in BET_COLS:
        df[f'{col}_decimal_odds'] = np.where(
            df[col] > 0, (df[col] / 100) + 1, (100 / abs(df[col])) + 1
        )

        df[f'{col}_implied_probability'] = 1 / df[f'{col}_decimal_odds']

    prob_cols = [c for c in df.columns if '_implied_probability' in c]
    df['median_probability'] = df[prob_cols].median(axis=1)
    df['mean_probability'] = df[prob_cols].mean(axis=1)

    cols_to_drop = (
        BET_COLS
        + [c for c in df.columns if '_decimal_odds' in c]
        + [c for c in df.columns if '_implied_probability' in c]
    )

    df = df.drop(columns=cols_to_drop)

    return df


def opponent_check(df: pd.DataFrame) -> pd.DataFrame:
    df['unique_id'] = df['blue_fighter'] + df['red_fighter'] + df['date'].astype(str)
    counts = (
        df.groupby('unique_id')
        .count()[['Bet']]
        .reset_index()
        .rename(columns={'Bet': 'count'})
    )
    counts = counts[counts['count'] == 2]
    df = df.merge(counts[['unique_id']], how='inner', on='unique_id')

    return df


def get_opponent_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    red_bets = df[df['red_fighter'] == df['Bet']].rename(
        columns={
            'mean_probability': 'red_mean_probability',
            'median_probability': 'red_median_probability',
        }
    )
    blue_bets = df[df['blue_fighter'] == df['Bet']].rename(
        columns={
            'mean_probability': 'blue_mean_probability',
            'median_probability': 'blue_median_probability',
        }
    )

    df = df.merge(
        red_bets[['unique_id', 'red_mean_probability', 'red_median_probability']]
    )
    df = df.merge(
        blue_bets[['unique_id', 'blue_mean_probability', 'blue_median_probability']]
    )

    df['opponent_mean_probability'] = np.where(
        df['fighter'] == 'red', df['blue_mean_probability'], df['red_mean_probability']
    )
    df['opponent_median_probability'] = np.where(
        df['fighter'] == 'red',
        df['blue_median_probability'],
        df['red_median_probability'],
    )

    df[MEAN_PROB] = df['mean_probability'] / (
        df['mean_probability'] + df['opponent_mean_probability']
    )
    df[MEDIAN_PROB] = df['median_probability'] / (
        df['median_probability'] + df['opponent_median_probability']
    )

    df['normalized_opponent_mean_probability'] = df['opponent_mean_probability'] / (
        df['mean_probability'] + df['opponent_mean_probability']
    )
    df['normalized_opponent_median_probability'] = df['opponent_median_probability'] / (
        df['median_probability'] + df['opponent_median_probability']
    )

    return df


def get_agg_win_data(
    df: pd.DataFrame, bucket_spacing: float, column: str
) -> pd.DataFrame:
    buckets = np.arange(0, 1.1, bucket_spacing)
    df['estimated_win_probability'] = pd.cut(df[column], buckets)
    df['estimated_win_probability'] = df['estimated_win_probability'].apply(
        lambda x: x.mid
    )

    agg_by = 'mean' if 'mean' in column else 'median'
    agg = (
        df.groupby('estimated_win_probability')
        .agg({'won': agg_by, column: 'count'})
        .reset_index()
    )
    agg = agg.rename(
        columns={
            column: 'count',
            'won': 'true_win_rate',
        }
    )
    agg = agg[agg['count'] > 0]

    return agg


def append_predictions(
    df: pd.DataFrame, linear_model: LinearRegression, column: str
) -> pd.DataFrame:
    df['raw_prediction'] = linear_model.intercept_ + linear_model.coef_ * df[column]
    if 'mean' in column:
        opponent_column = 'normalized_opponent_mean_probability'
    elif 'median' in column:
        opponent_column = 'normalized_opponent_median_probability'
    else:
        raise ValueError('Invalid column')

    df['opponent_raw_prediction'] = (
        linear_model.intercept_ + linear_model.coef_ * df[opponent_column]
    )
    df['prediction'] = df['raw_prediction'] / (
        df['raw_prediction'] + df['opponent_raw_prediction']
    )

    return df
