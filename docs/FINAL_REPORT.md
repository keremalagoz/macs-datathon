# Final Rapor — BTK Datathon 2025

Tarih: 31.08.2025
Takım: MACS

## Özet
- Amaç: Her oturum için `session_value` tahmini (MSE).
- En iyi LB: ~1307 (XGB full-train v2, log1p + backoff + TE). Son submission ~1800.
- Kısıtlar: Windows üzerinde paket sürüm farkları, GPU belirsizliği, sınırlı submission hakkı.

## Yaklaşım
1. Proje iskeleti ve GitHub kurulumu yapıldı; ham veriler `data/raw/` altına yerleştirildi.
2. Baseline ve minimal özellikler (v1) geliştirildi; ardından zengin oturum özellikleri (v2) eklendi.
3. Modelleme:
   - GroupKFold(user_id) ile sızıntısız CV.
   - Hedef dönüşümü: log1p; outlier etkisini azaltmak için winsorization (p=0.995) denendi.
   - Kullanıcı backoff ve son ürün/kategori target encoding (fold içi ve full-train).
   - Backend: XGBoost (tercih), SK HistGradientBoosting (fallback), LGBM (opsiyonel).
4. Hatalar ve düzeltmeler:
   - Test tahminlerinin fold-ortalaması hatası giderildi.
   - XGBoost API farklılıkları (eval_metric, callbacks) için uyarlamalar.
   - sample_submission’da user_id olmadığı için merge anahtarları düzeltildi (user_session).

## Sonuçlar (seçilmiş)
- Baseline global mean: LB ~4210
- SK-HGBR full-train v2 (log1p): LB ~1393
- XGB full-train v2 (log1p + backoff + TE): En iyi LB ~1307
- Blend (SK+XGB): ~1326 (kötüleşti)
- XGB winsorized full-train (final): ~1800

## Öğrenimler
- Ağır kuyruk (heavy tail) için log1p faydalı; winsorization bazı koşullarda generalizasyonu bozabiliyor.
- CV ve LB sapması: Özellikle kullanıcı/oturum dağılımındaki farklar ve son ürün etkisi.
- Stabil fallback’ler (SK-HGBR) kritik; GPU’ya güvenmek yerine CPU-hist yeterliydi.

## Reprodüksiyon
- Özellik üretimi:
  - scripts/features_v2_session.py → data/processed/session_features_v2_{train,test}.csv
- Eğitim/Submission (final):
  - scripts/model_final_fulltrain.py
  - Önerilen çalışma (Windows cmd):
    - set PYTHONUNBUFFERED=1
    - set TRY_GPU=0
    - set WINSORIZE_P=0.995
    - set N_ESTIMATORS=20000
    - python -u scripts\model_final_fulltrain.py
  - Çıktı: submissions/final_fulltrain_xgb_or_sk_v2_log_backoff_te_win0.995.csv

## Dosyalar
- scripts/baseline_v0_mean.py — basit ortalama
- scripts/features_v1_minimal.py — minimal oturum özellikleri
- scripts/features_v2_session.py — zengin v2 özellikleri
- scripts/model_lgbm_v1.py — çoklu backend (LGBM/XGB/SK), CV ve full-train
- scripts/model_xgb_cv_final.py — XGB 10-fold x 3-seed CV (gelişmiş log/progress)
- scripts/model_final_fulltrain.py — final tek-aşama eğitim ve submission
- scripts/blend_submissions.py — submission blend aracı
- scripts/quick_eda.py — ağır kuyruk analizi

## Sonraki Adımlar (gelecek çalışmalar)
- Katboost/NGBoost ve Huber/Fair loss ile robust hedefler
- Feature store: ürün/kategori geçmişine dayalı zaman pencereli istatistikler
- Daha güçlü target encoding (KFold-TE, smoothing) ve leave-one-user-out şemaları
- Seed bagging + OOF kalibrasyon

Teşekkürler.
