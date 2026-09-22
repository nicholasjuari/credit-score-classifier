"""
model_pipeline.py
=================
Komponen OOP untuk membangun, melatih, dan mengevaluasi model machine learning.

Dua class utama:
  - Trainer    : membangun pipeline (preprocessing + classifier) lalu melatihnya.
  - Evaluator  : menghitung metrik evaluasi pada data test.

Pipeline yang dibangun Trainer sepenuhnya berbasis scikit-learn Pipeline, jadi
bisa di-pickle sebagai satu objek dan di-load kembali untuk inference tanpa
rebuild manual.

Author: Nicholas Juari (2802413064)
"""
from __future__ import annotations
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class Trainer:
    """
    Membangun dan melatih pipeline lengkap untuk satu jenis model.

    Contoh:
        trainer = Trainer('rf', numeric_features, categorical_features,
                          n_estimators=200, max_depth=30)
        trainer.build()
        trainer.fit(X_train, y_train)
        model = trainer.pipeline
    """

    def __init__(self, model_name: str, numeric_features: list,
                 categorical_features: list, **clf_kwargs):
        if model_name not in ('logreg', 'rf', 'xgb'):
            raise ValueError(
                f"Unknown model_name: {model_name}. Use 'logreg', 'rf', or 'xgb'.")
        self.model_name = model_name
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.clf_kwargs = clf_kwargs
        self.pipeline: Pipeline | None = None

    # ---------- preprocessing ----------
    def build_preprocessor(self) -> ColumnTransformer:
        """ColumnTransformer: impute+scale untuk numerik, impute+onehot untuk kategori."""
        return ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]), self.numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore',
                                             sparse_output=False)),
                ]), self.categorical_features),
            ],
            remainder='drop'
        )

    # ---------- classifier ----------
    def _build_classifier(self):
        kw = dict(self.clf_kwargs)
        if self.model_name == 'logreg':
            return LogisticRegression(
                max_iter=kw.pop('max_iter', 500),
                class_weight=kw.pop('class_weight', 'balanced'),
                random_state=RANDOM_STATE, **kw)
        if self.model_name == 'rf':
            return RandomForestClassifier(
                n_estimators=kw.pop('n_estimators', 300),
                class_weight=kw.pop('class_weight', 'balanced'),
                random_state=RANDOM_STATE, n_jobs=-1, **kw)
        # xgb
        return XGBClassifier(
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **kw)

    # ---------- build + train ----------
    def build(self) -> Pipeline:
        """Rangkai preprocessor + classifier menjadi satu Pipeline."""
        self.pipeline = Pipeline([
            ('preprocess', self.build_preprocessor()),
            ('clf', self._build_classifier()),
        ])
        return self.pipeline

    def fit(self, X_train, y_train) -> Pipeline:
        """Latih pipeline. Otomatis build() dulu bila belum."""
        if self.pipeline is None:
            self.build()
        logger.info(f"Training {self.model_name}...")
        self.pipeline.fit(X_train, y_train)
        return self.pipeline


class Evaluator:
    """
    Menghitung metrik evaluasi model pada data test.

    Contoh:
        evaluator = Evaluator(label_encoder)
        metrics = evaluator.evaluate(model, X_test, y_test)
    """

    def __init__(self, le_target):
        self.le_target = le_target

    def evaluate(self, model, X_test, y_test) -> dict:
        """Hitung accuracy, macro/weighted F1, per-class F1, ROC-AUC, confusion matrix."""
        from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                      classification_report, confusion_matrix)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'roc_auc_ovr_macro': float(roc_auc_score(
                y_test, y_proba, multi_class='ovr', average='macro')),
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.le_target.classes_,
                output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }


# ---------------------------------------------------------------------------
# Wrapper fungsi (backward-compatible) untuk kode lama.
# ---------------------------------------------------------------------------
def build_preprocessor(numeric_features, categorical_features):
    return Trainer('rf', numeric_features, categorical_features).build_preprocessor()


def build_pipeline(model_name, numeric_features, categorical_features, **clf_kwargs):
    return Trainer(model_name, numeric_features, categorical_features,
                   **clf_kwargs).build()


def evaluate(model, X_test, y_test, le_target):
    return Evaluator(le_target).evaluate(model, X_test, y_test)
