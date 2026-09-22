"""
aws_train_pipeline.py
=====================
Training pipeline yang dirancang untuk dijalankan di AWS (EC2 atau SageMaker
Training Job). Bedanya dengan train_pipeline.py lokal:
  - Membaca dataset dari S3
  - Menyimpan artifacts (model, encoder, metadata) ke S3
  - Logging metrics ke CloudWatch (via Python logging — diteruskan ke
    CloudWatch Logs jika dijalankan di EC2/SageMaker dengan agent)

Usage:
    # Set environment variables atau gunakan ~/.aws/credentials
    export AWS_REGION=ap-southeast-1
    export S3_BUCKET=credit-score-2802413064
    export S3_DATA_KEY=data/data_C.csv
    export S3_ARTIFACT_PREFIX=artifacts/

    python aws_train_pipeline.py --model rf --n_estimators 200 --max_depth 30

Author: 2802413064
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Boto3 untuk S3 I/O (opsional kalau jalan di SageMaker built-in)
try:
    import boto3
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

# Reuse logic dari pipeline lokal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from data_processing import Preprocessor  # noqa: E402
from model_pipeline import Trainer, Evaluator  # noqa: E402

log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger('aws_train_pipeline')


# ----- S3 helpers -----
def s3_download(bucket: str, key: str, local_path: str):
    """Download file dari S3 ke local."""
    if not HAS_BOTO:
        raise RuntimeError("boto3 tidak terpasang. pip install boto3")
    s3 = boto3.client('s3')
    logger.info(f"Downloading s3://{bucket}/{key} -> {local_path}")
    s3.download_file(bucket, key, local_path)


def s3_upload(local_path: str, bucket: str, key: str):
    """Upload file lokal ke S3."""
    if not HAS_BOTO:
        raise RuntimeError("boto3 tidak terpasang. pip install boto3")
    s3 = boto3.client('s3')
    logger.info(f"Uploading {local_path} -> s3://{bucket}/{key}")
    s3.upload_file(local_path, bucket, key)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='rf', choices=['logreg', 'rf', 'xgb'])
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--n_estimators', type=int, default=None)
    p.add_argument('--max_depth', type=int, default=None)
    p.add_argument('--learning_rate', type=float, default=None)
    # AWS config — bisa via CLI atau env var
    p.add_argument('--s3_bucket', type=str, default=os.environ.get('S3_BUCKET'))
    p.add_argument('--s3_data_key', type=str,
                    default=os.environ.get('S3_DATA_KEY', 'data/data_C.csv'))
    p.add_argument('--s3_artifact_prefix', type=str,
                    default=os.environ.get('S3_ARTIFACT_PREFIX', 'artifacts/'))
    return p.parse_args()


def main():
    args = parse_args()
    if not args.s3_bucket:
        raise SystemExit("S3_BUCKET wajib di-set (via --s3_bucket atau env var)")

    run_id = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"=== AWS Training Run: {run_id} ===")
    logger.info(f"S3 source: s3://{args.s3_bucket}/{args.s3_data_key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_csv = os.path.join(tmpdir, 'data.csv')
        s3_download(args.s3_bucket, args.s3_data_key, local_csv)

        df_raw = pd.read_csv(local_csv)
        logger.info(f"Loaded {len(df_raw)} rows")

        # Cleaning
        preprocessor = Preprocessor(target_col='Credit_Score')
        df_clean = preprocessor.clean(df_raw, use_panel_imputation=True)
        income_cap = float(df_raw['Annual_Income'].astype(str)
                            .str.replace('_', '', regex=False)
                            .pipe(pd.to_numeric, errors='coerce').quantile(0.995))

        # Target encoding
        le_target = LabelEncoder()
        df_clean['Credit_Score_enc'] = le_target.fit_transform(df_clean['Credit_Score'])
        y = df_clean['Credit_Score_enc']
        groups = df_clean['Customer_ID']
        X = df_clean.drop(columns=['Credit_Score', 'Credit_Score_enc',
                                    'Customer_ID', 'Month'])
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        # Group-aware split
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                 random_state=args.random_state)
        tr, te = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Build & train
        clf_kwargs = {}
        if args.n_estimators and args.model in ('rf', 'xgb'):
            clf_kwargs['n_estimators'] = args.n_estimators
        if args.max_depth and args.model in ('rf', 'xgb'):
            clf_kwargs['max_depth'] = args.max_depth
        if args.learning_rate and args.model == 'xgb':
            clf_kwargs['learning_rate'] = args.learning_rate

        trainer = Trainer(args.model, numeric_features,
                          categorical_features, **clf_kwargs)
        trainer.build()
        logger.info(f"Training {args.model} dengan params={clf_kwargs}")
        pipeline = trainer.fit(X_train, y_train)

        # Evaluate
        evaluator = Evaluator(le_target)
        metrics = evaluator.evaluate(pipeline, X_test, y_test)
        logger.info(f"[METRIC] accuracy={metrics['accuracy']:.4f}")
        logger.info(f"[METRIC] f1_macro={metrics['f1_macro']:.4f}")
        logger.info(f"[METRIC] roc_auc_macro={metrics['roc_auc_ovr_macro']:.4f}")

        # Save artifacts lokal & upload ke S3
        model_path = os.path.join(tmpdir, 'best_model.pkl')
        encoder_path = os.path.join(tmpdir, 'label_encoder.pkl')
        meta_path = os.path.join(tmpdir, 'metadata.pkl')
        report_path = os.path.join(tmpdir, 'evaluation_report.json')

        joblib.dump(pipeline, model_path, compress=3)
        joblib.dump(le_target, encoder_path)
        metadata = {
            'best_model_name': args.model,
            'best_params': clf_kwargs,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'target_classes': le_target.classes_.tolist(),
            'test_metrics': {k: metrics[k] for k in ('accuracy', 'f1_macro',
                                                      'roc_auc_ovr_macro')},
            'income_cap_p99_5': income_cap,
            'trained_at': datetime.now().isoformat(),
            'aws_run_id': run_id,
        }
        joblib.dump(metadata, meta_path)
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        prefix = args.s3_artifact_prefix.rstrip('/') + '/'
        s3_upload(model_path, args.s3_bucket, prefix + 'best_model.pkl')
        s3_upload(encoder_path, args.s3_bucket, prefix + 'label_encoder.pkl')
        s3_upload(meta_path, args.s3_bucket, prefix + 'metadata.pkl')
        s3_upload(report_path, args.s3_bucket,
                  prefix + f'reports/eval_{run_id}.json')

        logger.info(f"=== Selesai. Artifacts di s3://{args.s3_bucket}/{prefix} ===")


if __name__ == '__main__':
    main()
