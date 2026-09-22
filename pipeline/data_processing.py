"""
data_processing.py
==================
Modul cleaning + feature engineering Dataset C (Credit Score Classification),
berbasis OOP. Class utama: Preprocessor.

Dipakai oleh:
  - pipeline/train_pipeline.py  (training & retraining)
  - deployment/inference.py     (prediction)

Author: Nicholas Juari (2802413064)
"""
from __future__ import annotations
import re
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Menangani seluruh proses pembersihan data dan feature engineering.

    Dirancang agar konsisten dipakai di training maupun inference: parameter
    seperti income_cap bisa di-pass dari hasil training supaya tidak terjadi
    kebocoran data saat prediksi.

    Contoh:
        prep = Preprocessor()
        df_clean = prep.clean(df_raw)
        X, y, groups = prep.split_features_target(df_clean)
    """

    # Konfigurasi domain (dipisah agar mudah di-tweak tanpa menyentuh logic)
    DROP_IDENTIFIERS = ['Unnamed: 0', 'ID', 'Name', 'SSN']
    DROP_AFTER_SPLIT = ['Customer_ID', 'Month']  # tidak masuk model

    NUMERIC_DIRTY_COLUMNS = [
        'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]

    PLACEHOLDER_MAP = {
        'Occupation': '_______',
        'Credit_Mix': '_',
        'Payment_Behaviour': '!@9#%8',
        'Payment_of_Min_Amount': 'NM',
    }

    OUTLIER_RULES = {
        # column: (min_valid, max_valid, mode)
        # mode='nan' -> nilai di luar rentang diganti NaN
        # mode='cap' -> nilai di atas max di-cap ke max
        'Age': (14, 100, 'nan'),
        'Num_Bank_Accounts': (0, 15, 'cap'),
        'Num_Credit_Card': (0, 15, 'cap'),
        'Interest_Rate': (0, 40, 'cap'),
        'Num_of_Loan': (0, 15, 'nan'),
        'Num_of_Delayed_Payment': (0, 30, 'cap'),
        'Num_Credit_Inquiries': (0, 30, 'cap'),
    }

    def __init__(self, target_col: str = 'Credit_Score'):
        self.target_col = target_col
        # income_cap disimpan setelah fit pertama, supaya bisa dipakai ulang
        self.income_cap_ = None

    # ---------- helper ----------
    @staticmethod
    def _parse_credit_history_age(x):
        """Convert 'X Years and Y Months' -> total bulan (int)."""
        if pd.isna(x):
            return np.nan
        m = re.match(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?', str(x))
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        return np.nan

    # ---------- main API ----------
    def clean(self, df_in: pd.DataFrame,
              income_cap: float | None = None,
              use_panel_imputation: bool = True) -> pd.DataFrame:
        """
        Lakukan semua step cleaning yang reproducible.

        Parameters
        ----------
        df_in : pd.DataFrame
            Raw dataframe dari CSV.
        income_cap : float, optional
            Cap untuk Annual_Income (biasanya P99.5 dari train set). Jika None,
            dihitung dari df_in lalu disimpan ke self.income_cap_.
        use_panel_imputation : bool
            Jika True, forward/backward fill per Customer_ID. Set False untuk
            inference single-row (tidak ada panel).
        """
        df = df_in.copy()
        logger.info(f"Mulai cleaning: shape input {df.shape}")

        # 1. Drop identifier
        df = df.drop(columns=[c for c in self.DROP_IDENTIFIERS if c in df.columns])

        # 2. Bersihkan kolom numerik (hapus underscore, ubah ke numeric)
        for c in self.NUMERIC_DIRTY_COLUMNS:
            if c in df.columns and df[c].dtype == object:
                df[c] = (df[c].astype(str).str.replace('_', '', regex=False)
                         .replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan}))
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 3. Parse Credit_History_Age -> total bulan
        if 'Credit_History_Age' in df.columns:
            df['Credit_History_Age'] = df['Credit_History_Age'].apply(
                self._parse_credit_history_age)

        # 4. Replace placeholder dengan NaN
        for col, placeholder in self.PLACEHOLDER_MAP.items():
            if col in df.columns:
                df[col] = df[col].replace(placeholder, np.nan)

        # 5. Outlier handling
        for col, (lo, hi, mode) in self.OUTLIER_RULES.items():
            if col not in df.columns:
                continue
            if mode == 'nan':
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
            elif mode == 'cap':
                df.loc[df[col] < lo, col] = lo
                df.loc[df[col] > hi, col] = hi

        # Annual_Income cap (pakai nilai yang di-pass / tersimpan jika ada)
        if 'Annual_Income' in df.columns:
            if income_cap is None:
                income_cap = self.income_cap_
            if income_cap is None:
                income_cap = df['Annual_Income'].quantile(0.995)
                logger.info(f"income_cap dihitung dari data: {income_cap:.2f}")
            self.income_cap_ = income_cap
            df.loc[df['Annual_Income'] > income_cap, 'Annual_Income'] = income_cap

        # 6. Feature engineering: jumlah jenis loan
        if 'Type_of_Loan' in df.columns:
            df['Num_Loan_Types'] = (df['Type_of_Loan'].fillna('').str.split(',').str.len()
                                    .where(df['Type_of_Loan'].notna(), 0))
            df = df.drop(columns=['Type_of_Loan'])

        # 7. Imputasi panel (forward/backward fill per Customer_ID)
        if use_panel_imputation and 'Customer_ID' in df.columns:
            df = df.sort_values(['Customer_ID', 'Month']
                                if 'Month' in df.columns else ['Customer_ID'])
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                df[col] = df.groupby('Customer_ID')[col].transform(
                    lambda s: s.ffill().bfill())
            cat_cols = [c for c in df.select_dtypes(include=['object']).columns
                        if c not in ('Customer_ID', self.target_col, 'Month')]
            for col in cat_cols:
                df[col] = df.groupby('Customer_ID')[col].transform(
                    lambda s: s.ffill().bfill())

        logger.info(f"Selesai cleaning: shape output {df.shape}")
        return df.reset_index(drop=True)

    def split_features_target(self, df_clean: pd.DataFrame):
        """Split DataFrame hasil cleaning menjadi X, y, groups (untuk GroupShuffleSplit)."""
        if self.target_col not in df_clean.columns:
            raise ValueError(f"Kolom target {self.target_col} tidak ditemukan.")
        groups = df_clean['Customer_ID'] if 'Customer_ID' in df_clean.columns else None
        y = df_clean[self.target_col]
        drop_cols = [self.target_col] + [c for c in self.DROP_AFTER_SPLIT
                                         if c in df_clean.columns]
        X = df_clean.drop(columns=drop_cols)
        return X, y, groups


# ---------------------------------------------------------------------------
# Wrapper fungsi (backward-compatible) supaya kode lama yang meng-import
# clean_dataframe / split_features_target tetap berjalan.
# ---------------------------------------------------------------------------
def clean_dataframe(df_in, income_cap=None, use_panel_imputation=True):
    return Preprocessor().clean(df_in, income_cap=income_cap,
                                use_panel_imputation=use_panel_imputation)


def split_features_target(df_clean, target_col='Credit_Score'):
    return Preprocessor(target_col=target_col).split_features_target(df_clean)
