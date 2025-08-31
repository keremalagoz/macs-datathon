# Nasıl Çalıştırılır

Gereksinimler: Python 3.10+, pip, Windows cmd.

## 1) Veri
- data/raw/train.csv
- data/raw/test.csv
- data/raw/sample_submission.csv

## 2) Özellikler
- v2 özelliklerini üretmek için (gerekliyse):
  - python scripts\features_v2_session.py

## 3) Modeller
- Hızlı baseline:
  - python scripts\baseline_v0_mean.py

- Final full-train (önerilen, CPU):
  - set PYTHONUNBUFFERED=1
  - set TRY_GPU=0
  - set WINSORIZE_P=0.995
  - set N_ESTIMATORS=20000
  - python -u scripts\model_final_fulltrain.py

- Alternatif CV modeli (ileri seviye):
  - set PYTHONUNBUFFERED=1
  - set TRY_GPU=0
  - set MAX_FOLDS=10
  - set MAX_SEEDS=3
  - python -u scripts\model_xgb_cv_final.py

## 4) Çıktı
- submissions/ klasörüne CSV yazılır; Kaggle’a doğrudan yüklenebilir.
