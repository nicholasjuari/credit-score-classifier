"""
aws_inference.py
================
Inferencing dengan model yang artifacts-nya tersimpan di S3.
Dipakai oleh Streamlit yang di-deploy di EC2.

Cara kerja:
  1. Saat pertama kali instantiate, cek apakah artifact ada di /tmp.
  2. Kalau tidak ada (atau --force_refresh), download dari S3.
  3. Load model & predict.

Author: 2802413064
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import boto3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'pipeline'))
sys.path.insert(0, str(PROJECT_ROOT / 'deployment'))
from data_processing import Preprocessor  # noqa: E402

logger = logging.getLogger(__name__)

LOCAL_CACHE = Path('/tmp/credit_score_artifacts')
S3_BUCKET = os.environ.get('S3_BUCKET', 'credit-score-2802413064')
S3_ARTIFACT_PREFIX = os.environ.get('S3_ARTIFACT_PREFIX', 'artifacts/').rstrip('/') + '/'


def download_artifacts(force: bool = False):
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client('s3')
    for name in ('best_model.pkl', 'label_encoder.pkl', 'metadata.pkl'):
        local = LOCAL_CACHE / name
        if local.exists() and not force:
            continue
        key = S3_ARTIFACT_PREFIX + name
        logger.info(f"Downloading s3://{S3_BUCKET}/{key} -> {local}")
        s3.download_file(S3_BUCKET, key, str(local))


class CreditScorePredictorS3:
    """Versi predictor yang muat artifact dari S3 (cache lokal di /tmp)."""

    def __init__(self, force_refresh: bool = False):
        download_artifacts(force=force_refresh)
        self.model = joblib.load(LOCAL_CACHE / 'best_model.pkl')
        self.label_encoder = joblib.load(LOCAL_CACHE / 'label_encoder.pkl')
        self.metadata = joblib.load(LOCAL_CACHE / 'metadata.pkl')
        logger.info(f"Loaded {self.metadata['best_model_name']} from S3")

    def predict(self, input_data):
        single = isinstance(input_data, dict)
        raw = pd.DataFrame([input_data]) if single else input_data.copy()
        cap = self.metadata.get('income_cap_p99_5')
        cleaned = Preprocessor().clean(raw, income_cap=cap, use_panel_imputation=False)
        drop_cols = ['Credit_Score', 'Customer_ID', 'Month']
        cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns])
        expected = (self.metadata['numeric_features']
                    + self.metadata['categorical_features'])
        for col in expected:
            if col not in cleaned.columns:
                cleaned[col] = np.nan
        X = cleaned[expected]
        proba = self.model.predict_proba(X)
        pred = self.model.predict(X)
        classes = self.label_encoder.classes_
        out = []
        for i in range(len(X)):
            out.append({
                'predicted_class': str(classes[pred[i]]),
                'probabilities': {str(c): float(p) for c, p in zip(classes, proba[i])},
                'confidence': float(proba[i].max()),
            })
        return out[0] if single else out
