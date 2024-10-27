import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score

from constants import Y_PRED, Y_PRED, OUTPUT_DIR


def produce_histogram(df: pd.DataFrame, column: str):
    wins = df[df['won'] == 1]
    losses = df[df['won'] == 0]

    plt.figure(figsize=(10, 6))
    plt.hist(wins[column], color='red', alpha=0.5, label='Winner', bins=20)
    plt.hist(losses[column], color='blue', alpha=0.5, label='Loser', bins=20)
    plt.xlabel('Win Probability')
    plt.ylabel('Count of Fighters')
    plt.title('Probability by Results')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}probability_by_results_{column}.png')
    plt.close()


def produce_linear_win_rate_plot(win_data: pd.DataFrame, column: str):

    lm = LinearRegression()
    X = win_data['estimated_win_probability'].values.reshape(-1, 1)
    y = win_data['true_win_rate']
    lm.fit(X, y, sample_weight=win_data['count'])
    r2 = lm.score(X, y, sample_weight=win_data['count'])

    plt.figure(figsize=(10, 6))
    plt.plot(X, lm.predict(X), alpha=0.5)
    plt.scatter(
        win_data['estimated_win_probability'],
        win_data['true_win_rate'],
        s=win_data['count'] / 10,
        color='black',
    )

    equation = f'y = {lm.coef_[0]:.3f}x + {lm.intercept_:.3f}'
    text = f'{equation}\nR² = {r2:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
    )

    plt.xlabel('Estimated Win Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Estimated Probability vs Win Rate')

    plt.savefig(f'{OUTPUT_DIR}win_prob_vs_rate_{column}.png')
    plt.close()

    return lm


def produce_pr_curve(y_true: pd.Series, y_pred: pd.Series, column: str):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aucpr_value = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Model')
    plt.ylim(0.4, 1)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random')
    plt.legend()

    text = f'AUC-PR: {aucpr_value: .3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(
        0.05,
        0.1,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.savefig(f'{OUTPUT_DIR}precision_recall_curve_{column}.png')
    plt.close()


def produce_roc_curve(y_true: pd.Series, y_pred: pd.Series, column: str):
    roc_value = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    text = f'AUC-ROC: {roc_value: .3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(
        0.8,
        0.1,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
    )

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.plot(fpr, tpr, label='Model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    plt.savefig(f'{OUTPUT_DIR}roc_curve_{column}.png')
    plt.close()
