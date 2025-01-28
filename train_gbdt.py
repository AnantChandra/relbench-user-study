import argparse
import os
import numpy as np
import time

import duckdb
from relbench.tasks import get_task
from torch_frame import TaskType, stype
from torch_frame.gbdt import LightGBM, XGBoost
from torch_frame.data import Dataset
from torch_frame.typing import Metric

from inferred_stypes import task_to_stypes
import utils

SEED = 42
DATASET_TO_DB = {
    'rel-stack': 'stack/stack.db',
    'rel-amazon': 'amazon/amazon.db',
    'rel-hm': 'hm/hm.db',
    'rel-f1': 'f1/f1.db',
    'rel-event': 'event/event.db',
}
TASK_PARAMS = {
    'rel-stack-user-engagement': {
        'dir': 'stack/user-engagement',
        'target_col': 'contribution',
        'table_prefix': 'user_engagement',
        'identifier_cols': ['OwnerUserId', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-stack-user-badge': {
        'dir': 'stack/user-badge',
        'target_col': 'WillGetBadge',
        'table_prefix': 'user_badge',
        'identifier_cols': ['UserId', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-stack-post-votes': {
        'dir': 'stack/post-votes',
        'target_col': 'popularity',
        'table_prefix': 'post_votes',
        'identifier_cols': ['PostId', 'timestamp'],
        'tune_metric': Metric.MAE,
        'task_type': TaskType.REGRESSION,
    },
    'rel-amazon-user-churn': {
        'dir': 'amazon/user-churn',
        'target_col': 'churn',
        'table_prefix': 'user_churn',
        'identifier_cols': ['customer_id', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-amazon-user-ltv': {
        'dir': 'amazon/user-ltv',
        'target_col': 'ltv',
        'table_prefix': 'user_ltv',
        'identifier_cols': ['customer_id', 'timestamp'],
        'tune_metric': Metric.MAE,
        'task_type': TaskType.REGRESSION,
    },
    'rel-amazon-item-churn': {
        'dir': 'amazon/item-churn',
        'target_col': 'churn',
        'table_prefix': 'item_churn',
        'identifier_cols': ['product_id', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-amazon-item-ltv': {
        'dir': 'amazon/item-ltv',
        'target_col': 'ltv',
        'table_prefix': 'item_ltv',
        'identifier_cols': ['product_id', 'timestamp'],
        'tune_metric': Metric.MAE,
        'task_type': TaskType.REGRESSION,
    },
    'rel-hm-item-sales': {
        'dir': 'hm/item-sales',
        'target_col': 'sales',
        'table_prefix': 'item_sales',
        'identifier_cols': ['article_id', 'timestamp'],
        'tune_metric': Metric.MAE,
        'task_type': TaskType.REGRESSION,
    },
    'rel-hm-user-churn': {
        'dir': 'hm/user-churn',
        'target_col': 'churn',
        'table_prefix': 'user_churn',
        'identifier_cols': ['customer_id', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-f1-driver-position': {
        'dir': 'f1/driver-position',
        'target_col': 'position',
        'table_prefix': 'driver_position',
        'identifier_cols': ['driverId', 'date'],
        'tune_metric': Metric.MAE,
        'task_type': TaskType.REGRESSION,
    },
    'rel-f1-driver-dnf': {
        'dir': 'f1/driver-dnf',
        'target_col': 'did_not_finish',
        'table_prefix': 'driver_dnf',
        'identifier_cols': ['driverId', 'date'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-f1-driver-top3': {
        'dir': 'f1/driver-top3',
        'target_col': 'qualifying',
        'table_prefix': 'driver_top3',
        'identifier_cols': ['driverId', 'date'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-event-user-repeat': {
        'dir': 'event/user-repeat',
        'target_col': 'target',
        'table_prefix': 'user_repeat',
        'identifier_cols': ['user', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-event-user-ignore': {
        'dir': 'event/user-ignore',
        'target_col': 'target',
        'table_prefix': 'user_ignore',
        'identifier_cols': ['user', 'timestamp'],
        'tune_metric': Metric.ROCAUC,
        'task_type': TaskType.BINARY_CLASSIFICATION,
    },
    'rel-event-user-attendance': {
        'dir': 'event/user-attendance',
        'target_col': 'target',
        'table_prefix': 'user_attendance',
        'identifier_cols': ['user', 'timestamp'],
        'tune_metric': Metric.MAE,
        'task_type': TaskType.REGRESSION,
    },
}
NUM_TRIALS = 10


def get_matching_rows(feats_df, labels_df, identifier_cols, n_rows=1000):
    """Get exactly matching rows between features and labels dataframes."""
    # Convert identifier columns to strings first
    feat_str_df = feats_df[identifier_cols].astype(str)
    label_str_df = labels_df[identifier_cols].astype(str)
    
    # Create match keys
    feat_keys = feat_str_df.agg('-'.join, axis=1)
    label_keys = label_str_df.agg('-'.join, axis=1)
    
    # Find common indices
    common_keys = set(feat_keys) & set(label_keys)
    if len(common_keys) < n_rows:
        raise ValueError(f'Only found {len(common_keys)} matching rows, needed {n_rows}')
        
    # Get first n_rows of matches
    selected_keys = list(common_keys)[:n_rows]
    
    # Filter and sort both dataframes
    matched_feats = feats_df[feat_keys.isin(selected_keys)]
    matched_labels = labels_df[label_keys.isin(selected_keys)]
    
    # Ensure same order in both dataframes
    matched_feats['_key'] = feat_keys[matched_feats.index]
    matched_labels['_key'] = label_keys[matched_labels.index]
    
    matched_feats = matched_feats.sort_values('_key').drop('_key', axis=1)
    matched_labels = matched_labels.sort_values('_key').drop('_key', axis=1)
    
    return matched_feats.reset_index(drop=True), matched_labels.reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--dataset', '-d', type=str, help='Relbench dataset name')
    parser.add_argument('--task', '-t', type=str, help='Relbench task name')
    parser.add_argument('--booster', '-b', type=str, default='lgbm', help='One of "xgb" or "lgbm"')
    parser.add_argument('--subsample', '-s', type=int, default=0,
                        help=(
                            'If provided, use a subset of the training set to speed up training. '
                            'If generate_feats is set, features will only be generated for this '
                            'subset. '
                        ))
    parser.add_argument('--generate_feats', action='store_true',
                        help='Whether to (re)generate features specified in feats.sql')
    parser.add_argument('--drop_cols', nargs='+', default=[], help='Columns to drop')
    args = parser.parse_args()
    
    print('Initializing...')
    full_task_name = f'{args.dataset}-{args.task}'
    task_params = TASK_PARAMS[full_task_name]
    
    # Load task data first
    print('Loading task data...')
    task = get_task(args.dataset, args.task, download=True)
    val_task_df = task.get_table("val").df
    test_task_df = task.get_table("test").df
    
    # Connect and load feature data
    print('Loading feature data...')
    conn = duckdb.connect(DATASET_TO_DB[args.dataset])
    
    if args.generate_feats:
        print('Generating features.')
        start = time.time()
        with open(os.path.join(task_params['dir'], 'feats.sql')) as f:
            template = f.read()
        for s in ['train', 'val', 'test']:
            print(f'Creating {s} table')
            query = utils.render_jinja_sql(template, dict(set=s, subsample=args.subsample))
            conn.sql(query)
            print(f'{s} table created')
        print(f'Features generated in {time.time() - start:,.0f} seconds.')

    train_df = conn.sql(f'select * from {task_params["table_prefix"]}_train_feats').df()
    val_df = conn.sql(f'select * from {task_params["table_prefix"]}_val_feats').df()
    test_df = conn.sql(f'select * from {task_params["table_prefix"]}_test_feats').df()
    conn.close()

    # Get matching rows for validation and test sets
    print('Finding matching rows...')
    val_df, val_task_df = get_matching_rows(
        val_df, val_task_df, task_params['identifier_cols'], n_rows=1000
    )
    test_df, test_task_df = get_matching_rows(
        test_df, test_task_df, task_params['identifier_cols'], n_rows=1000
    )
    
    # Process columns
    col_to_stype = task_to_stypes[full_task_name]
    drop_cols = task_params['identifier_cols'] + args.drop_cols
    
    train_df = train_df.drop(args.drop_cols, axis=1)
    val_df = val_df.drop(args.drop_cols, axis=1)
    test_df = test_df.drop(args.drop_cols, axis=1)
    
    for col in args.drop_cols:
        del col_to_stype[col]
        
    if args.subsample > 0 and not args.generate_feats:
        train_df = train_df.head(args.subsample)

    print('Materializing torch-frame dataset.')
    start = time.time()
    
    for k, v in col_to_stype.items():
        if v == stype.text_embedded:
            raise NotImplementedError(
                'Embeddings for text columns not supported for speed considerations. Either drop '
                'them with the --drop_cols flag or see relbench/examples for how to use embeddings.'
            )

    train_dset = Dataset(
        train_df,
        col_to_stype=col_to_stype,
        target_col=task_params['target_col'],
    ).materialize()
    
    val_tf = train_dset.convert_to_tensor_frame(val_df)
    test_tf = train_dset.convert_to_tensor_frame(test_df)
    
    print(f'Materialized torch-frame dataset in {time.time() - start:,.0f} seconds.')
    print(f'Train Size: {train_dset.tensor_frame.num_rows:,} x {train_dset.tensor_frame.num_cols:,}')
    
    # Initialize model
    booster = LightGBM if args.booster == 'lgbm' else XGBoost
    if task_params['task_type'] == TaskType.BINARY_CLASSIFICATION:
        gbdt = booster(task_params['task_type'], num_classes=2, metric=task_params['tune_metric'])
    elif task_params['task_type'] == TaskType.REGRESSION:
        gbdt = booster(task_params['task_type'], metric=task_params['tune_metric'])
    
    # Train and tune
    print('Starting hparam tuning.')
    start = time.time()
    gbdt.tune(tf_train=train_dset.tensor_frame, tf_val=val_tf, num_trials=NUM_TRIALS)
    print(f'Hparam tuning completed in {time.time() - start:,.0f} seconds.')
    
    model_path = os.path.join(task_params['dir'], f'{full_task_name}_{args.booster}.json')
    print(f'Saving model to "{model_path}".')
    gbdt.save(model_path)
    print()

    # Evaluate
    print('Evaluating model.')
    
    # Get predictions
    val_pred = gbdt.predict(tf_test=val_tf).numpy()
    test_pred = gbdt.predict(tf_test=test_tf).numpy()
    
    # Get task tables
    val_table = task.get_table("val")
    test_table = task.get_table("test")
    
    # Update the task tables with our matched subset
    val_table.df = val_task_df
    test_table.df = test_task_df
    
    # Evaluate
    print(f'Val: {task.evaluate(val_pred, val_table)}')
    print()
    print(f'Test: {task.evaluate(test_pred, test_table)}')
