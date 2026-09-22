"""
inference.py
============
Skrip inferencing untuk model Credit Score Classification.

Bisa dipakai dengan dua cara:
  1. Sebagai modul Python (import CreditScorePredictor di Streamlit / API)
  2. Sebagai CLI script: python inference.py --input sample.json

Author: 2802413064
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

# Tambahkan path agar bisa import data_processing dari folder pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'pipeline'))
from data_processing import Preprocessor  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'


class CreditScorePredictor:
    """
    Wrapper untuk prediksi credit score dari raw input
    (single dict atau DataFrame multi-row).

    Usage:
        predictor = CreditScorePredictor()
        result = predictor.predict({'Age': 35, 'Annual_Income': 50000, ...})
        print(result)
        # {'predicted_class': 'Standard', 'probabilities': {'Good': ..., 'Standard': ..., 'Poor': ...}}
    """

    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.label_encoder = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = self.artifact_dir / 'best_model.pkl'
        encoder_path = self.artifact_dir / 'label_encoder.pkl'
        meta_path = self.artifact_dir / 'metadata.pkl'
        for p in (model_path, encoder_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Artifact tidak ditemukan: {p}. Jalankan train_pipeline.py dulu.")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.metadata = joblib.load(meta_path)
        logger.info(f"Loaded model: {self.metadata['best_model_name']}")

    def _prepare_input(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Bersihkan raw input dengan logic yang SAMA dengan training,
        kecuali tanpa panel imputation (karena inference biasanya single row).

        Untuk single row, kolom yang missing setelah cleaning akan di-handle
        oleh SimpleImputer di dalam pipeline.
        """
        income_cap = self.metadata.get('income_cap_p99_5')
        preprocessor = Preprocessor()
        cleaned = preprocessor.clean(raw, income_cap=income_cap,
                                     use_panel_imputation=False)
        # Drop kolom yang tidak masuk model (kalau ada)
        drop_cols = ['Credit_Score', 'Customer_ID', 'Month']
        cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns])
        # Pastikan semua fitur yang model harapkan ada (isi None kalau missing)
        expected = (self.metadata['numeric_features']
                    + self.metadata['categorical_features'])
        for col in expected:
            if col not in cleaned.columns:
                cleaned[col] = np.nan
        return cleaned[expected]  # urutkan sesuai expected

    def predict(self, input_data) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Prediksi credit score.

        Parameters
        ----------
        input_data : dict atau pd.DataFrame
            Raw input — kolom mengikuti format CSV training.
        """
        single = False
        if isinstance(input_data, dict):
            raw = pd.DataFrame([input_data])
            single = True
        elif isinstance(input_data, pd.DataFrame):
            raw = input_data.copy()
        else:
            raise TypeError("input_data harus dict atau pd.DataFrame")

        X = self._prepare_input(raw)
        proba = self.model.predict_proba(X)
        pred = self.model.predict(X)
        classes = self.label_encoder.classes_

        results = []
        for i in range(len(X)):
            results.append({
                'predicted_class': str(classes[pred[i]]),
                'probabilities': {str(c): float(p) for c, p in zip(classes, proba[i])},
                'confidence': float(proba[i].max()),
            })
        return results[0] if single else results


# ----- CLI entrypoint -----
def main():
    p = argparse.ArgumentParser(description='Credit Score inference CLI')
    p.add_argument('--input', type=str, required=True,
                   help='Path ke JSON file (single object atau array of objects)')
    p.add_argument('--artifact_dir', type=str, default=str(DEFAULT_ARTIFACT_DIR))
    args = p.parse_args()

    with open(args.input) as f:
        payload = json.load(f)

    predictor = CreditScorePredictor(artifact_dir=Path(args.artifact_dir))
    result = predictor.predict(payload)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
