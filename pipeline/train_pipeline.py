"""
train_pipeline.py
=================
Pipeline retraining end-to-end:
  1. Load data dari path yang dikonfigurasi
  2. Clean & feature engineering
  3. Group-aware train/test split
  4. Train model (configurable via CLI)
  5. Log experiment ke MLflow (params, metrics, model artifact)
  6. Save model terbaik ke artifacts/

Usage:
  python train_pipeline.py --data ../data_C.csv --model rf
  python train_pipeline.py --data ../data_C.csv --model xgb --n_estimators 500 --max_depth 8
  python train_pipeline.py --data ../data_C.csv --model logreg

MLflow UI:
  mlflow ui --backend-store-uri file:./mlruns
  buka http://localhost:5000

Author: 2802413064
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

# Tambah parent path agar import data_processing & model_pipeline berfungsi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_processing import Preprocessor  # noqa: E402
from model_pipeline import Trainer, Evaluator  # noqa: E402

# ----- Logging -----
log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger('train_pipeline')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
ARTIFACT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Credit Score retraining pipeline')
    p.add_argument('--data', type=str, default=str(PROJECT_ROOT / 'data_C.csv'),
                   help='Path ke CSV input')
    p.add_argument('--model', type=str, default='rf', choices=['logreg', 'rf', 'xgb'])
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--experiment_name', type=str, default='credit-score-classification')
    p.add_argument('--run_name', type=str, default=None)
    # Hyperparams umum (akan di-forward ke classifier)
    p.add_argument('--n_estimators', type=int, default=None)
    p.add_argument('--max_depth', type=int, default=None)
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--C', type=float, default=None, help='Regularization untuk LogReg')
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # MLflow setup — pakai folder lokal mlruns/
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        logger.info(f"=== Pipeline run: {run_name} ===")

        # 1) Load data
        logger.info(f"Loading data from {args.data}")
        df_raw = pd.read_csv(args.data)
        mlflow.log_param('data_path', args.data)
        mlflow.log_param('rows_input', len(df_raw))

        # 2) Cleaning (pakai class Preprocessor - OOP)
        preprocessor = Preprocessor(target_col='Credit_Score')
        df_clean = preprocessor.clean(df_raw, use_panel_imputation=True)
        mlflow.log_param('rows_after_clean', len(df_clean))
        # Simpan income_cap yang dipakai (dari raw — di production seharusnya dari train slice)
        income_cap_value = float(df_raw['Annual_Income'].astype(str)
                                  .str.replace('_', '', regex=False)
                                  .pipe(pd.to_numeric, errors='coerce').quantile(0.995))
        mlflow.log_param('income_cap_p99_5', income_cap_value)

        # 3) Target encoding + split features
        le_target = LabelEncoder()
        df_clean['Credit_Score_enc'] = le_target.fit_transform(df_clean['Credit_Score'])
        y_full = df_clean['Credit_Score_enc']
        groups_full = df_clean['Customer_ID']
        X_full = df_clean.drop(columns=['Credit_Score', 'Credit_Score_enc',
                                         'Customer_ID', 'Month'])

        numeric_features = X_full.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_full.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Numeric features: {len(numeric_features)}, "
                    f"Categorical: {len(categorical_features)}")

        # 4) Group-aware split
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                 random_state=args.random_state)
        train_idx, test_idx = next(gss.split(X_full, y_full, groups=groups_full))
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        # 5) Build pipeline dengan hyperparam dari CLI (filter None)
        clf_kwargs = {}
        if args.n_estimators is not None and args.model in ('rf', 'xgb'):
            clf_kwargs['n_estimators'] = args.n_estimators
        if args.max_depth is not None and args.model in ('rf', 'xgb'):
            clf_kwargs['max_depth'] = args.max_depth
        if args.learning_rate is not None and args.model == 'xgb':
            clf_kwargs['learning_rate'] = args.learning_rate
        if args.C is not None and args.model == 'logreg':
            clf_kwargs['C'] = args.C

        mlflow.log_param('model_name', args.model)
        for k, v in clf_kwargs.items():
            mlflow.log_param(k, v)

        # Build + train pakai class Trainer (OOP)
        trainer = Trainer(args.model, numeric_features, categorical_features,
                          **clf_kwargs)
        pipeline = trainer.build()

        # 6) Train
        pipeline = trainer.fit(X_train, y_train)

        # 7) Evaluate pakai class Evaluator (OOP)
        evaluator = Evaluator(le_target)
        metrics = evaluator.evaluate(pipeline, X_test, y_test)
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test f1_macro: {metrics['f1_macro']:.4f}")
        logger.info(f"Test ROC-AUC (macro): {metrics['roc_auc_ovr_macro']:.4f}")

        # Log metrics yang scalar
        for k in ('accuracy', 'f1_macro', 'f1_weighted', 'roc_auc_ovr_macro'):
            mlflow.log_metric(k, metrics[k])
        # Per-class F1
        for cls_name in le_target.classes_:
            mlflow.log_metric(f"f1_{cls_name}",
                              metrics['classification_report'][cls_name]['f1-score'])

        # 8) Simpan artifacts lokal
        model_path = ARTIFACT_DIR / 'best_model.pkl'
        encoder_path = ARTIFACT_DIR / 'label_encoder.pkl'
        meta_path = ARTIFACT_DIR / 'metadata.pkl'
        report_path = ARTIFACT_DIR / 'evaluation_report.json'

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
            'income_cap_p99_5': income_cap_value,
            'trained_at': datetime.now().isoformat(),
        }
        joblib.dump(metadata, meta_path)
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Artifacts disimpan di {ARTIFACT_DIR}")

        # 9) Log ke MLflow
        # log_model bisa gagal di MLflow 3.x karena validasi 'skops' yang ketat
        # (UntrustedTypesFoundException: numpy.dtype). Fallback: log file .pkl
        # langsung sebagai artifact, sehingga model tetap tersimpan di run.
        model_path = ARTIFACT_DIR / 'best_model.pkl'
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # Coba dengan trusted types (MLflow >= 2.16 / 3.x)
                    mlflow.sklearn.log_model(
                        pipeline, name='model',
                        skops_trusted_types=['numpy.dtype'])
                except TypeError:
                    # MLflow lama tidak punya argumen tsb
                    mlflow.sklearn.log_model(pipeline, artifact_path='model')
        except Exception as e:
            logger.warning(f"log_model dilewati ({type(e).__name__}); "
                           f"model tetap di-log sebagai artifact .pkl")
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path='model')

        mlflow.log_artifact(str(encoder_path))
        mlflow.log_artifact(str(meta_path))
        mlflow.log_artifact(str(report_path))

        logger.info("=== Pipeline selesai ===")
        return metrics


if __name__ == '__main__':
    main()
