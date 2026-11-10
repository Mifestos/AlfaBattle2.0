# Alfa Battle 2.0: Credit Default Prediction

Решение задачи бинарной классификации для предсказания вероятности дефолта клиента на основе транзакционной активности.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow)

## Результаты

Финальный пайплайн использует ансамбль из трех моделей (Blending).

| Модель | OOF ROC-AUC |
| :--- | :--- |
| LightGBM (Baseline) | 0.76067 |
| CatBoost | 0.76070 |
| Transformer (DL) | 0.72762 |
| **3-Way Blend (Final)** | **0.76373** |

## Стек 
Pandas, NumPy, PyArrow, LightGBM, CatBoost, Scikit-learn, PyTorch, SciPy.
## Ключевые технические решения 

* **Leakage-Free OOF Target Encoding:** Кодирование высококардинальных категориальных признаков (MCC-коды, тип продукта) со строгим разделением по фолдам и сглаживанием (Smoothing). Исключает утечку таргета (Data Leakage).
* **Multimodal Deep Learning (PyTorch):** 
  * *Sequence Tower:* GRU / Transformer Encoder с Attention и позиционным кодированием для сырых последовательностей транзакций.
  * *Tabular Tower:* FT-Transformer для обработки плотных табличных признаков.
  * *Fusion:* Cross-Attention механизм для объединения временных и табличных модальностей.
* **Ensemble Architecture (Blending):** Взвешенное усреднение трех независимых моделей (LightGBM + CatBoost + DL) с подбором оптимальных весов через Grid Search.
* **Modular Pipeline Orchestration:** Разделение пайплайна на изолированные модули (`v01` — FE+LGBM, `v02` — DL, `v03` — CatBoost, `v04` — Ensemble). Единый CLI-оркестратор (`run.py`), строгая типизация и изолированное логирование.

## Структура проекта

```text
.
├── v01_baseline_gbdt/       # FE + LightGBM
├── v02_deep_learning/       # PyTorch (Dataset, Model, Train)
├── v03_catboost/            # CatBoost (Oblivious Trees)
├── v04_ensemble/            # Блендинг (Grid Search) и Inference
├── artifacts/               # Модели (OOF, weights, parquet)
└── run.py                   # Оркестратор (CLI)
```
## Запуск
```bash
python run.py
```
## Автор
GitHub: @Mifestos