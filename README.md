# Final Project — Model Deployment
**Nicholas Juari — 2802413064**
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nicholasjuari-credit-score.streamlit.app)

## Deskripsi
Sistem klasifikasi credit score nasabah (Good / Standard / Poor) memakai dataset C.
Setelah membandingkan Logistic Regression, Random Forest, dan XGBoost, model
yang dipakai adalah **Random Forest** (test accuracy 70.3%, F1 macro 0.69, ROC-AUC 0.85).

Project ini di-deploy dua cara: lokal dengan Streamlit, dan di cloud AWS (S3 + EC2).

## Demo

Live app: https://nicholasjuari-credit-score.streamlit.app

GitHub repo: https://github.com/nicholasjuari/credit-score-classifier

## Struktur Project

```
2802413064/
├── notebook/
│   └── EDA_and_Modelling.ipynb     EDA + modelling
│
├── pipeline/                       Pipeline training lokal + MLflow tracking
│   ├── data_processing.py          Cleaning + feature engineering
│   ├── model_pipeline.py           Build pipeline (preprocessor + classifier)
│   └── train_pipeline.py           Entry-point training
│
├── deployment/                     Deployment lokal (Streamlit)
│   ├── app.py                      UI Streamlit
│   ├── inference.py                Class CreditScorePredictor
│   └── requirements.txt            Dependencies
│
├── aws_deployment/                 Deployment AWS (S3 + EC2)
│   ├── aws_train_pipeline.py       Training pipeline (baca/tulis S3)
│   ├── aws_inference.py            Inference dengan model dari S3
│   ├── app_aws.py                  Streamlit untuk EC2
│   └── architecture_diagram.png
│
├── artifacts/                      Model artifacts
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   └── metadata.pkl
│
└── docs/
    ├── LOCAL_VS_CLOUD_COMPARISON.pdf   Perbandingan lokal vs cloud
    ├── Documentation_AWS.pdf           Screenshot test case deployment AWS
    └── VIDEO_LINK.pdf                  Link video penjelasan
```

## Cara Menjalankan

**Lokal:**
```bash
pip install -r deployment/requirements.txt
streamlit run deployment/app.py
```

**Training ulang** (parameter bisa diganti dari command line):
```bash
python pipeline/train_pipeline.py --model rf --n_estimators 200 --max_depth 30
```

**AWS:** Streamlit jalan di EC2 dengan model di-load dari S3. Langkah lengkapnya
ada di kode pada folder `aws_deployment/`.

## Catatan
Dataset (`data_C.csv`) tidak disertakan karena ukurannya besar dan sudah tersedia
di sisi pengajar. Untuk menjalankan training lokal, taruh file dataset di root folder
project (sejajar dengan `pipeline/`).
