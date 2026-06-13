# MA-CETSP — Session Context

## Proje Nedir?

**MA-CETSP** (Memetic Algorithm for Clustered Euclidean Travelling Salesman Problem) — bir optimizasyon algoritması araştırması. Temel fikir: evrimsel algoritmada her nesilde üretilen "offspring" çözümleri yerel aramaya (VND) sokmadan önce, **survival analysis ML modelleri** kullanarak kalitesiz olanları filtrele, böylece VND zamanı tasarrufu yap.

Eski makale (LA-CETSP, Lei & Hao 2024): 3 model (COX, RSF, GBSA), 4 instance, sabit eğitim zamanı.

**Yeni katkı (bu çalışma):**
- 9 model: COX, RSF, GBSA, DEEPSURV, SSVM, WEIBULLAFT, KNN, ELASTICNET, MTLR
- 68 instance (4 grup: varied, or2/10/30, rdmRad, real-world car_door)
- Island parallelism: 9 island aynı anda çalışır, her island farklı modeli kullanır
- Adaptive training trigger: sabit iter=250 yerine stagnation + event count + budget fractions
- Adaptive threshold: validation set üzerinde percentile search (50-95) ile optimize edilir
- **Rolling rejection-rate cap** (yeni eklendi): production drift sorununu çözer

---

## Mevcut Durum

### Neden 1000 iter koşuyoruz?

**Keşfedilen sorun:** COX ve SSVM modelleri bazı seed'lerde bonus1000 instance'ında **%40+ rejection rate** yapıyordu. Bu durum search'ü mahvediyordu (iter=449'da takılıp kalıyordu, sıfır gelişme). Nedeni:
- Val-set'te kalibre edilen threshold, production offline'a geçince shift oluyor (covariate drift)
- Bazı seed'lerin training datası "over-rejecting" model yetiştiriyor

**Uygulanan fix (Rolling Rejection-Rate Cap):**
`src/Genetic/Population.cpp` + `include/Defs.hpp` + `include/Genetic/Population.hpp` değiştirildi.
Her 50 ML-eligible offspring'den sonra rejection rate hesaplanır; %30'u geçerse bir sonraki window için filter **suspend** edilir. Böylece search diversity'sini korur.

```
ML_ROLLING_WINDOW          = 50   // window uzunluğu
ML_MAX_ROLLING_REJECT_RATE = 0.30 // bu geçilirse filter suspend
```

**Mevcut koşu:** Fix uygulandıktan sonra **15 development instance × 10 seed × 9 model = 1350 ML run + 150 BASELINE = 1500 satır**, 1000 iter, doğrulama amaçlı. Executable: `build/Release/MA-CETSP.exe`.

### Mevcut analiz sonuçları (rolling cap fix ONCESI, re-run henuz yapilmadi)

Rolling cap fix uygulandı ama 15 development instance henüz yeni binary ile koşulmadı.
Aşağıdaki tablo eski binary'nin sonuçlarıdır. **En güncel analiz → `analysis/results/model_comparison.csv`**

```
Model        Deg%    Speedup   Rej%   Wins
BASELINE      ---     1.000    0.0     77 (51.3%)
RSF          0.20%    0.926   19.4%    ...
GBSA         0.28%    1.077   20.6%    <- en iyi ML
DEEPSURV     0.25%    0.950   15.8%
COX          1.02%    1.418   26.6%    <- bonus1000'de felaket
SSVM         1.09%    1.403   30.3%    <- en kotu
```

Wilcoxon: RSF p=0.0597, GBSA p=0.0869, DEEPSURV p=0.1005 (hepsi p>0.05, non-significant at n=15)

---

## Sonraki Adımlar

### 1. Dev re-run bittikten sonra analiz

```powershell
# Logları parse et
python analysis/parse_logs.py

# Analiz çalıştır
python analysis/analyze_development.py
```

**Beklentiler:**
- COX/SSVM: degradation %1+ → %0.3-0.4% düşmeli (cap düzeltirse)
- Speedup korunmalı (cap çok sık tetiklenirse kaybolur)
- RSF/GBSA/DEEPSURV: değişmemeli (bunların zaten sorunu yoktu)

**Senaryolar:**
- Senaryo A (iyi): cap tetikleniyor, COX deg %0.3-0.4'e düşüyor, speedup korunuyor
- Senaryo B (nötr): cap çok sık, ML neredeyse çalışmıyor, speedup kayboluyor
- Senaryo C (kötü): cap + kötü model = her ikisi de kötü

### 2. Final test set (asıl paper verisi)

Dev analizinden sonra **68 instance × 10 seed × 9 model, 5000 iter** koşulacak.
Birden fazla PC'ye bölüştürülecek. Her PC ayrı instance set'i çalıştırır, sonunda
`solutions/experiment_results/` klasörleri birleştirilir, tek parse.

Dev set instances (daha önce koşuldu, 1000 iter):
`0,5,11,16,24,27,36,39,44,50,51,61,62,64,66`

Test set instances (53 instance, 5000 iter):
`1,2,3,4,6,7,8,9,10,12,13,14,15,17,18,19,20,21,22,23,25,26,28,29,30,31,32,33,34,35,37,38,40,41,42,43,45,46,47,48,49,52,53,54,55,56,57,58,59,60,63,65,67`

Heavy instances (1000 node, çok uzun sürer): 28 (dsj1000_or2), 35 (dsj1000_or10), 42 (dsj1000_or30), 48 (bonus1000rdmRad)
→ PC'lere eşit dağıtılmalı.

---

## Analiz Pipeline

### Dosya yapısı

```
solutions/experiment_results/
  instance<N>_<timestamp>/
    run<R>_seed<S>.log     <- C++ ciktisi, tum island loglari burada
    summary.txt

analysis/
  parse_logs.py          ← logları parse eder, CSV üretir
  analyze_development.py ← analiz ve tablolar

analysis/parsed/         ← parse_logs.py çıktısı
  island_summaries.csv   ← ana tablo: instance × run × seed × island × model
  training_events.csv    ← ne zaman, hangi koşulda training tetiklendi
  thresholds.csv         ← her island'ın threshold ve percentile değeri
  convergence.csv        ← iter bazında best cost (her 10 iter)
  rejection_timeline.csv ← 50-iter window'da kaç reject
  rejections.csv         ← bireysel reject eventları

analysis/results/        ← analyze_development.py çıktısı
  model_comparison.csv   ← Deg%, Speedup, Rej%, Wins
  group_breakdown.csv    ← instance grup bazında aynı metrikler
  best_iter_analysis.csv ← convergence hızı
  threshold_stability.csv← threshold std, pct_mean, obj_mean
  wilcoxon_test.csv      ← her model vs BASELINE p-value
  training_events_summary.csv
```

### Çalıştırma komutları

```powershell
# Experiment başlat (örnek)
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 `
  -Instances 0,5,11,16,24 -Islands 10 -Iterations 1000

# Parse
python analysis/parse_logs.py

# Analiz
python analysis/analyze_development.py
```

---

## Kritik Teknik Notlar

### Thread-unsafe cout (çözüldü)
C++ parallel island'lar `std::cout`'a mutex olmadan yazıyor → SUMMARY satırları bozuluyor.
`parse_logs.py`'de `RE_PARALLEL_ROW` fallback mekanizması var: eğer bir island için SUMMARY
parse edilemezse, log sonundaki parallel summary tablosundan (bozulmamış) alınır.
**Bu olmadan 1500 satırın ~10'u eksik kalıyordu.**

### Adaptive training trigger
```
Defs.hpp sabitler:
  ML_MIN_EVENTS        = 100   // ≥100 death event
  ML_TRAIN_FRAC_MIN    = 0.20  // en erken: iter budget'ın %20'si
  ML_TRAIN_FRAC_MAX    = 0.25  // soft ceiling
  ML_TRAIN_FRAC_HARD   = 0.50  // hard fallback
  ML_PATIENCE_FRACTION = 0.40  // patience budget'ın %40'ı tükenmeli
```

### Threshold calibration
`ml/scripts/threshold_utils.py`: val set üzerinde percentile 50-95 arasında search yapar.
Objective = `rejection_rate × survival_gap`. En iyi percentile → full data'ya uygulanır.
**CANDIDATE_PERCENTILES = range(50, 96, 5) — 70-95'e daraltmayın, RSF/GBSA zarar görür.**

### Rolling cap (yeni)
`include/Defs.hpp`: `ML_ROLLING_WINDOW=50`, `ML_MAX_ROLLING_REJECT_RATE=0.30`
`include/Genetic/Population.hpp`: `ml_win_attempts_`, `ml_win_rejects_`, `ml_suspended_`
Her 50 offspring'de window reset. Reject rate > %30 → `ml_suspended_=true` → bir window boş geç.

---

## Paper Durumu

Deadline: **18 Haziran 2026** (hedef: 16 Haziran). Şu an yazılabilir: Introduction, Related Work, Methodology.
Results/Discussion için test set (5000 iter) bekleniyor.

Yeni eklenecek 6 model için citation lazım:
- DeepSurv: Katzman et al. (2018), BMC Med Inform Decis Mak
- SSVM: Van Belle et al. (2011), Bioinformatics
- WeibullAFT: Carroll (2003) veya scikit-survival dökümantasyonu
- MTLR: Fotso (2018), arXiv
- KNN survival: Altman (1992), Stat Med
- ElasticNet Cox: Simon et al. (2011), J Stat Softw
