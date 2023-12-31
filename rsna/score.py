import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission.min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    label_group_losses = []
    label_group_precisions = {}
    label_group_recalls = {}
    label_group_aucs = {}
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )
        
        label_group_precisions[category] = sklearn.metrics.precision_score(
            y_true=np.argmax(solution[col_group].values, axis=-1, keepdims=True),
            y_pred=np.argmax(submission[col_group].values, axis=-1, keepdims=True),
            average='weighted'
        )

        label_group_recalls[category] = sklearn.metrics.recall_score(
            y_true=np.argmax(solution[col_group].values, axis=-1, keepdims=True),
            y_pred=np.argmax(submission[col_group].values, axis=-1, keepdims=True),
            average='weighted'
        )

        label_group_aucs[category] = sklearn.metrics.roc_auc_score(
            y_true=solution[col_group].values,
            y_score=submission[col_group].values,
            multi_class='ovr'
        )

    print(label_group_precisions)
    print(label_group_recalls)
    print(label_group_aucs)
    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    any_injury_loss = sklearn.metrics.log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=solution['any_injury_weight'].values
    )

    label_group_losses.append(any_injury_loss)
    return np.mean(label_group_losses)


solution = pd.read_csv('solution.csv')
submission = pd.read_csv('submission.csv')
print(score(solution, submission, 'patient_id'))

# data = pd.read_csv('data/train.csv')
# data['bowel_weight'] = np.where(data['bowel_injury'] == 1, 2, 1)
# data['extravasation_weight'] = np.where(data['extravasation_injury'] == 1, 6, 1)
# data['kidney_weight'] = np.select([data['kidney_healthy'] == 1, data['kidney_low'] == 1, data['kidney_high'] == 1], [1, 2, 4])
# data['liver_weight'] = np.select([data['liver_healthy'] == 1, data['liver_low'] == 1, data['liver_high'] == 1], [1, 2, 4])
# data['spleen_weight'] = np.select([data['spleen_healthy'] == 1, data['spleen_low'] == 1, data['spleen_high'] == 1], [1, 2, 4])
# data['any_injury_weight'] = np.where((data['bowel_healthy'] == 1) & (data['extravasation_healthy'] == 1) &
#                               (data['kidney_healthy'] == 1) & (data['liver_healthy'] == 1) & (data['spleen_healthy'] == 1), 1, 6)
# data.to_csv('solution.csv')