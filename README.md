# BTK Datathon 2025 — Session Value Prediction

Bu repo, BTK Datathon 2025 (Kaggle) yarışması için uçtan uca ML pipeline çalışmalarını içerir. Hedef: test setindeki her session için `session_value` tahmini (MSE ile değerlendirilecek).

## Yapı

- `data/`
  - `raw/`       → Orijinal ham veri (train.csv, test.csv, sample_submission.csv)
  - `external/`  → Harici/yardımcı veri kaynakları
  - `interim/`   → Ara çıktı/temizlenmiş geçici dosyalar
  - `processed/` → Özellik mühendisliği sonrası model girdi setleri
- `notebooks/`
  - `eda/`, `cleaning/`, `features/`, `modelling/` → Keşif ve prototipleme defterleri
- `src/`         → Pipeline modülleri (ileride eklenecek)
- `scripts/`     → Komut satırı betikleri (etl, train, infer vb.)
- `configs/`     → Deney/parametre/config yaml dosyaları
- `experiments/` → Deney kayıtları (metrikler, artefakt yolları)
- `submissions/` → Kaggle submission dosyaları (csv)
- `reports/`     → Raporlar ve görseller (`reports/figures/`)
- `docs/`        → Ek dokümantasyon

## Roller
- Doğu → Data Cleaning (eksik veri, aykırı değer, dtype, leakage önleme)
- Furkan → Feature Engineering (session-level aggregations, historical stats, ratios)
- Kerem → Modelling (baseline, tuning, ensembling)

## Veri Yerleşimi
- Kaggle'dan indirilen dosyaları `data/raw/` içine koyun:
  - `data/raw/train.csv`
  - `data/raw/test.csv`
  - `data/raw/sample_submission.csv`

`.gitignore` ham ve türetilmiş büyük dosyaları takip dışı bırakır; dizinler `.gitkeep` ile versiyonlanır.

## Çalışma Akışı
EDA → Data Cleaning → Feature Engineering → Modeling → Ensemble → Submission

Leaderboard’a erken submission yapıp iteratif geliştirme hedeflenir.
