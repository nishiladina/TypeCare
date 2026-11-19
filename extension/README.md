# Log 

### 11/16
- set up `type4py-extension` branch
- downloaded `ManyTypes4Py` dataset
- downloaded `Type4Py` model weights 
- set up venv for conducting the extension
- swapped out `PYInfer` with `Type4Py` because `PYInfer` does not have open source code

### 11/17
- loaded the ONNX model, W2V embeddings, and Annoy clusters, extracted code features in `download_type4py_weights`
- `generate_type4py_predictions` script for baseline data
- Type4Py prediction verified, baseline set  

### 11/18
- updated `preanalysis/run.py` and `analysis/run.py` to include Type4Py handling
- `evaluation/eval.py` updated 
- ran smoke test on Type4Py + TypeCare

### 11/19
- Completed full pre-analysis (Pyright diagnostics) across all 2,687 test targets in `ManyTypes4Py`
- Ran TypeCare re-ranking on Type4Py candidates; generated `output/type4py_result.pkl`
- Whitelisted `output/type4py_result.pkl` and `prediction/Type4Py/transformed_result.json` in `.gitignore`
- Added standalone table viewer: `python extension/print_type4py_results.py`
- Also integrated with `analysis/run.py`: `python -m analysis.run --evaluate --tool type4py`

#### Evaluation Results Summary (ManyTypes4Py Benchmark, 2,687 Targets)

##### 1. Overall Top-K Accuracy (Exact Match)
| Rank | Type4Py Baseline | TypeCare + Type4Py | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Top 1** | 1,049 (39.0%) | **1,698 (63.2%)** | **+62.1%** |
| **Top 3** | 1,322 (49.2%) | **1,869 (69.6%)** | **+41.5%** |
| **Top 5** | 1,407 (52.4%) | **1,928 (71.8%)** | **+37.0%** |

##### 2. Base Top-K Accuracy (Base Type Match)
| Rank | Type4Py Baseline | TypeCare + Type4Py | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Top 1** | 1,196 (44.5%) | **1,730 (64.4%)** | **+44.7%** |
| **Top 3** | 1,610 (59.9%) | **1,966 (73.2%)** | **+22.2%** |
| **Top 5** | 1,750 (65.1%) | **2,053 (76.4%)** | **+17.4%** |

##### 3. Categorized Top-K Accuracy (Exact Match)
| Category | Metric | Type4Py Baseline | TypeCare + Type4Py | Relative Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Element** (1,404 targets) | Top 1 | 978 (69.6%) | **1,205 (85.8%)** | **+23.3%** |
| | Top 3 | 1,171 (83.3%) | **1,266 (90.1%)** | **+8.2%** |
| | Top 5 | 1,217 (86.6%) | **1,279 (91.0%)** | **+5.1%** |
| **Parameterized** (483 targets) | Top 1 | 45 (9.3%) | **170 (35.2%)** | **+278.5%** |
| | Top 3 | 106 (21.9%) | **212 (43.9%)** | **+100.5%** |
| | Top 5 | 129 (26.7%) | **229 (47.4%)** | **+77.5%** |
| **User-Defined** (800 targets) | Top 1 | 26 (3.3%) | **323 (40.4%)** | **+1124.2%** |
| | Top 3 | 45 (5.6%) | **391 (48.9%)** | **+773.2%** |
| | Top 5 | 61 (7.6%) | **420 (52.6%)** | **+592.1%** |

##### 4. Argument vs. Return Type Top-K Accuracy
| Type Slot | Metric | Type4Py Baseline | TypeCare + Type4Py | Relative Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Argument** (1,847 targets) | Top 1 | 737 (39.9%) | **1,219 (66.0%)** | **+65.4%** |
| | Top 3 | 935 (50.6%) | **1,346 (72.9%)** | **+44.1%** |
| | Top 5 | 992 (53.7%) | **1,388 (75.1%)** | **+39.9%** |
| **Return** (840 targets) | Top 1 | 305 (36.3%) | **416 (49.5%)** | **+36.4%** |
| | Top 3 | 375 (44.6%) | **443 (52.7%)** | **+18.2%** |
| | Top 5 | 397 (47.3%) | **450 (53.6%)** | **+13.3%** |

# Raw output below:
```
Top K
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  1049  |   39.0%    | 1698  |   63.2%   | 62.1% |
| Top 3 |  1322  |   49.2%    | 1869  |   69.6%   | 41.5% |
| Top 5 |  1407  |   52.4%    | 1928  |   71.8%   | 37.0% |
+-------+--------+------------+-------+-----------+-------+

Base Top K
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  1118  |   41.6%    | 1749  |   65.1%   | 56.5% |
| Top 3 |  1449  |   53.9%    | 1929  |   71.8%   | 33.2% |
| Top 5 |  1567  |   58.3%    | 1988  |   74.0%   | 26.9% |
+-------+--------+------------+-------+-----------+-------+

Element Problems
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  978   |   69.6%    | 1205  |   85.8%   | 23.3% |
| Top 3 |  1171  |   83.3%    | 1266  |   90.1%   | 8.2%  |
| Top 5 |  1217  |   86.6%    | 1279  |   91.0%   | 5.1%  |
+-------+--------+------------+-------+-----------+-------+

Parameterized Problems
+-------+--------+------------+-------+-----------+--------+
| Rank  | Before | Before (%) | After | After (%) |  Diff  |
+-------+--------+------------+-------+-----------+--------+
| Top 1 |   45   |    9.3%    |  170  |   35.2%   | 278.5% |
| Top 3 |  106   |   21.9%    |  212  |   43.9%   | 100.5% |
| Top 5 |  129   |   26.7%    |  229  |   47.4%   | 77.5%  |
+-------+--------+------------+-------+-----------+--------+

User Problems
+-------+--------+------------+-------+-----------+---------+
| Rank  | Before | Before (%) | After | After (%) |  Diff   |
+-------+--------+------------+-------+-----------+---------+
| Top 1 |   26   |    3.3%    |  323  |   40.4%   | 1124.2% |
| Top 3 |   45   |    5.6%    |  391  |   48.9%   | 773.2%  |
| Top 5 |   61   |    7.6%    |  420  |   52.6%   | 592.1%  |
+-------+--------+------------+-------+-----------+---------+

=== Argument and Return Type Top-K ===

Argument Type
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  742   |   40.2%    | 1223  |   66.2%   | 64.7% |
| Top 3 |  944   |   51.1%    | 1355  |   73.4%   | 43.6% |
| Top 5 |  1006  |   54.5%    | 1401  |   75.9%   | 39.3% |
+-------+--------+------------+-------+-----------+-------+

Return Type
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  307   |   36.5%    |  420  |   50.0%   | 37.0% |
| Top 3 |  378   |   45.0%    |  448  |   53.3%   | 18.4% |
| Top 5 |  401   |   47.7%    |  455  |   54.2%   | 13.6% |
+-------+--------+------------+-------+-----------+-------+

Total Type
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  1049  |   39.0%    | 1643  |   61.1%   | 56.7% |
| Top 3 |  1322  |   49.2%    | 1803  |   67.1%   | 36.4% |
| Top 5 |  1407  |   52.4%    | 1856  |   69.1%   | 31.9% |
+-------+--------+------------+-------+-----------+-------+
=== Argument and Return Type Base Top-K ===

Argument Type
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  782   |   42.3%    | 1283  |   69.5%   | 64.3% |
| Top 3 |  1022  |   55.3%    | 1428  |   77.3%   | 39.8% |
| Top 5 |  1110  |   60.1%    | 1467  |   79.4%   | 32.1% |
+-------+--------+------------+-------+-----------+-------+

Return Type
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  336   |   40.0%    |  466  |   55.5%   | 38.8% |
| Top 3 |  427   |   50.8%    |  501  |   59.6%   | 17.3% |
| Top 5 |  457   |   54.4%    |  521  |   62.0%   | 14.0% |
+-------+--------+------------+-------+-----------+-------+

Total Type
+-------+--------+------------+-------+-----------+-------+
| Rank  | Before | Before (%) | After | After (%) | Diff  |
+-------+--------+------------+-------+-----------+-------+
| Top 1 |  1118  |   41.6%    | 1749  |   65.1%   | 56.5% |
| Top 3 |  1449  |   53.9%    | 1929  |   71.8%   | 33.2% |
| Top 5 |  1567  |   58.3%    | 1988  |   74.0%   | 26.9% |
+-------+--------+------------+-------+-----------+-------+

Total Problem Number: 2687
Argument Problem Number: 1847
Return Problem Number: 840
```