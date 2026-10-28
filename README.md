# SEPSIS-6Q: a quantum kernel early warning benchmark for sepsis in intensive care

Team Kanban, Qiskit Fall Fest 2026, Quantum Machine Learning track.

SEPSIS-6Q asks whether a compact quantum kernel support vector machine (QSVM) on eight qubits detects more hours before sepsis onset than strong classical models, when every model sees the same eight vital sign features, the same small set of labelled training rows and the same tuning budget. The data are the public PhysioNet/Computing in Cardiology Challenge 2019 records from two hospital systems: one system is used for training, validation and an internal test, and the other is held out as an external test. All quantum results come from exact statevector simulation on a laptop class CPU (plus one binomial shot sampling study); nothing was run on quantum hardware.

<!-- BEGIN status -->
Status: PRELIMINARY. The numbers below come from training sizes [1000] and draw seeds [0] on 20,331 set A files and 6,486 set B files, before the full run. They will be replaced by the full run.
<!-- END status -->

<!-- BEGIN summary -->
Not available: the results files for this block are missing (TypeError).
<!-- END summary -->

## 1. Prespecified question

This question, the eight features, the operating points and the headline training size were fixed in this file and in `src/sepsis6q/config.py` before any model was trained.

> At the same false alarm rate, does a compact quantum kernel SVM detect more pre onset sepsis hours than the best classical model trained on the same features and the same training rows?

Primary metric: sensitivity (the fraction of positive, pre onset rows that raise an alarm) at exactly 5% false positive rate, read from the ROC curve of each test set. Headline training size: N = 2000 balanced rows, averaged over 3 random draws. The best matched classical model is chosen on validation data, not test data. An improvement is claimed only if the 95% patient level bootstrap interval of the paired difference excludes zero.

## 2. Problem and relevance

Sepsis is a dysregulated response to infection and a leading cause of death in intensive care. Treatment is time critical, so hospitals use early warning scores that raise an alarm before clinical recognition. Every early warning system trades missed cases against alarm fatigue: a system that alarms too often is ignored. This is why the comparison here is made at a fixed false alarm rate rather than by accuracy.

The intended outcome is a fair, reproducible answer to the question above, not a deployable tool. The setting deliberately mimics a hospital with few labelled sepsis cases, which is where a model with a different inductive bias could matter most. Local interest in machine learning early warning exists in the UAE: a cohort study at Cleveland Clinic Abu Dhabi with NYU Abu Dhabi built machine learning models for in hospital deterioration (Lashen et al., JMIR AI 2023). This project uses public ICU data from the United States, not UAE data. It is not validated for UAE use or for any clinical use, and it is a research prototype, not a medical device.

## 3. Data and preprocessing

### 3.1 Source

PhysioNet/Computing in Cardiology Challenge 2019 training data (Reyna et al., 2020; dataset DOI 10.13026/v64v-d857), sets A and B, which come from two separate hospital systems. Each patient file holds hourly rows with about 40 clinical columns and `SepsisLabel`. `scripts/download_data.py` fetches every file from the PhysioNet directory listing into `data/raw/setA/` and `data/raw/setB/`, which are git ignored. No raw or processed data are committed.

### 3.2 Label behaviour and the early warning task

In the challenge, `SepsisLabel` is 1 from six hours before clinical onset of sepsis onwards. The exploratory check in `sepsis6q.data.label_behaviour` confirms this per patient: the table below counts septic patients whose label is a single run of zeros followed by ones, patients whose label is already 1 in the first row, and the number of label 1 rows per septic patient.

Task definition. Onset is the first label 1 row plus six hours. For each septic patient, rows at or after onset are removed, because an alarm there is no longer an early warning. Positive rows are the label 1 rows before onset (at most six per patient). Negative rows are all earlier rows of septic patients and all rows of non septic patients.

<!-- BEGIN data -->
| Set | Patients | Hourly rows | Septic patients | Label monotone (0 then 1) | Label 1 from first row | Label 1 rows per septic patient (min / median / max) | Task rows | Positive task rows | Row prevalence |
|---|---|---|---|---|---|---|---|---|---|
| Set A | 20331 | 790106 | 1789 | 1789 of 1789 | 203 | 6 / 10 / 10 | 783714 | 10734 | 1.37% |
| Set B | 6486 | 245698 | 360 | 360 of 360 | 70 | 1 / 10 / 10 | 244445 | 2153 | 0.88% |

Set A patient split (all patients): 12198 training pool, 4066 validation, 4067 internal test.

| Partition | Patients | Septic patients | Rows | Positive rows | Row prevalence |
|---|---|---|---|---|---|
| Training pool (set A, 60%) | 12198 | 1073 | 471880 | 6438 | 1.36% |
| Validation subset (set A, 20% split) | 2272 | 200 | 88271 | 1200 | 1.36% |
| Internal test subset (set A, 20% split) | 3408 | 300 | 130212 | 1800 | 1.38% |
| External test subset (set B) | 5405 | 300 | 203541 | 1793 | 0.88% |
<!-- END data -->

### 3.3 Splits

All splits are by patient and seeded (seed 2026). Set A patients are split 60% training pool, 20% validation and 20% internal test, stratified by sepsis status. Set B, a different hospital system, is the external test. To keep runtime manageable, validation and both test sets are fixed random patient subsets, stratified so that the natural prevalence of septic patients is preserved: 200 septic patients for validation and 300 septic patients for each test set, plus the matching proportion of non septic patients. Validation and test rows stay at their natural prevalence.

Training rows for the matched comparison are drawn only from the training pool, in sizes N of 250, 500, 1000 and 2000, balanced 1:1 between positive and negative rows, with 3 random draws (seeds 0, 1, 2) per N. The same drawn rows go to every matched model.

### 3.4 Features (identical for every matched model)

Eight features were prespecified and never changed: the current values of HR, Resp, Temp, SBP, MAP and O2Sat, and the 6 hour change in HR and in Resp. Only past information is used. Vital signs are forward filled within a patient from past values only, with a carry limit of 12 hours. Remaining gaps, including trends in the first six hours, are imputed with medians of the training rows. Every feature is then scaled to [0, 1] with the minimum and maximum of the training rows, and validation and test values are clipped to that range. All transforms are fitted on the N training rows of each draw only. The quantum models multiply the scaled values by pi times a bandwidth parameter to obtain rotation angles.

`tests/test_data_pipeline.py` checks that features do not change when future rows are removed, that patient ids are disjoint across splits, that training draws come from the training pool only, and that transforms fitted on training rows are unaffected by changes to test rows.

## 4. Classical baselines

Matched models, all on the same eight features, the same N rows and the same preprocessing:

* Logistic regression (scikit-learn `LogisticRegression`).
* Support vector machine with an RBF kernel (scikit-learn `SVC`), tuning C and gamma. This is the closest classical counterpart of the QSVM: the same SVM solver with a classical kernel.
* Gradient boosted trees (scikit-learn `HistGradientBoostingClassifier`, small grid, no early stopping).

Tuning protocol, identical for every matched model including the QSVMs: patient grouped 3 fold cross validation on the N training rows (`GroupKFold`, so no patient contributes rows to both sides of a fold), selecting the configuration with the highest mean held out AUROC. Grid sizes are comparable across models:

<!-- BEGIN grids -->
| Model | Grid | Configurations |
|---|---|---|
| Logistic regression | C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] | 6 |
| RBF SVM | C in [0.1, 1.0, 10.0]; gamma in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] | 18 |
| Gradient boosted trees | learning_rate in [0.03, 0.1, 0.3]; max_depth in [2, 3, 4]; max_iter in [100, 300] | 18 |
| QSVM, ZZ entangling map | C in [0.1, 1.0, 10.0]; bandwidth in [0.25, 0.5, 1.0]; reps in [1, 2] | 18 |
| QSVM, product map (control) | C in [0.1, 1.0, 10.0]; bandwidth in [0.25, 0.5, 1.0]; reps in [1, 2] | 18 |
<!-- END grids -->

Reference models (context only, not a fair competitor because they see far more labelled rows): logistic regression with balanced class weights and gradient boosted trees (learning rate 0.1, depth 4, 300 iterations), both trained on every row of the training pool with the same eight features, at natural prevalence, with fixed configurations and no tuning. The wider feature set with missingness indicators was not run (see limitations).

Credit: the task and data come from the PhysioNet/CinC Challenge 2019 (Reyna et al., 2020). The classical models are standard scikit-learn implementations (Pedregosa et al., 2011). Gradient boosted trees on hourly vital signs were among the strong approaches in that challenge; our version is deliberately restricted to the same eight features as the quantum model so that the comparison is fair.

## 5. Quantum approach

### 5.1 Feature map and kernel

Each of the eight scaled features is encoded on one qubit as an angle x_i = b pi u_i, where u_i is in [0, 1] and b is a tuned bandwidth. One repetition of the entangling map is a Hadamard gate on every qubit, a phase gate P(2 x_i) on every qubit, and for each neighbouring pair (i, i+1) on a line the block CX, P(2 (pi minus x_i)(pi minus x_{i+1})), CX. This is the ZZFeatureMap construction of Havlicek et al. (2019) with linear entanglement, and the repetition count is tuned. The kernel is the state fidelity k(x, y) = |<psi(x)|psi(y)>|^2, which on hardware would be estimated by the compute uncompute circuit U(x) followed by the inverse of U(y), as the probability of measuring all zeros. The SVM is scikit-learn `SVC` with this precomputed kernel.

Control: the same QSVM with a product (non entangling) map, which keeps the Hadamard and single qubit phase gates and drops the pairwise blocks. It has the same grid as the entangling QSVM. If the entangling map does not beat the product map, the data do not support a claim that entanglement matters.

Tuning C, the bandwidth b and the repetition count is the fair counterpart of tuning C and gamma for the RBF SVM.

### 5.2 Implementation

Qiskit is the reference definition of the map (`sepsis6q.quantum.feature_map_circuit`). For speed, `sepsis6q.quantum.statevectors` is a batched numpy statevector simulator of the same map: since each repetition is a Hadamard layer followed by a diagonal phase, the state is computed with one product by the 256 by 256 Walsh Hadamard matrix and one elementwise phase per repetition. `tests/test_quantum.py` asserts that the numpy states equal Qiskit `Statevector` results to 1e-9 for random inputs, repetition counts 1 to 3 and both maps, and that the kernel equals the kernel of Qiskit's library `zz_feature_map` with linear entanglement. States are computed once per row, the training Gram matrix is one matrix product, and test decision values are computed in chunks against the support vectors only, never storing a full test by train matrix.

### 5.3 Hypothesis, limitation addressed and what is not claimed

Hypothesis: the entangling map encodes pairwise interactions between neighbouring vital signs (for example heart rate with respiratory rate, blood pressure with mean arterial pressure) as phases, which might give a useful inductive bias when labelled sepsis cases are scarce. The classical limitation it addresses is sample efficiency in the small data setting of a single hospital, not computational cost.

At eight qubits this kernel is easy to simulate classically: a state has 256 amplitudes and the whole study runs on a CPU. No computational speedup is claimed, and none is possible at this size. Any advantage under test is about inductive bias and sample efficiency.

## 6. Evaluation protocol

Operating points were fixed before looking at test data.

* Model comparison at equal false alarm rate (primary): sensitivity at exactly 5% false positive rate, read from each test set's ROC curve with linear interpolation between ROC points.
* Deployment style: the threshold is chosen on the validation set at 95% and at 90% specificity and then applied unchanged to each test set; the realised false alarm rate on the test set is reported next to sensitivity.
* AUROC, and AUPRC (average precision) with its prevalence baseline.
* Patient level detection at the deployment thresholds: the fraction of septic patients with at least one alarm in their label 1 window before onset, and false alarms per 100 non septic patient days (each row is one hour).
* Uncertainty: patient level bootstrap with 500 resamples of the test patients for every headline metric and for the paired differences. In each resample the metric of a model is the mean over its 3 training draws, and the same resampled patients are used for every model, so differences are paired.
* Learning curves: the primary metric against N for all matched models, mean and spread over draws, with the full pool reference models as horizontal lines.

Metric code is tested against scikit-learn (`tests/test_metrics.py`), including weighted cases used by the bootstrap.

## 7. Results

All tables and figures in this section are generated by `scripts/render_readme_tables.py` from the files in `results/`.

### 7.0 All runs (row per model, draw and evaluation set)

<!-- BEGIN runs -->
| Model | N | Draw seed | Set | Sensitivity at 5% FPR | AUROC | AUPRC | Prevalence |
|---|---|---|---|---|---|---|---|
| Logistic regression | 1000 | 0 | test_A | 0.127 | 0.607 | 0.022 | 0.014 |
| Logistic regression | 1000 | 0 | test_B | 0.149 | 0.608 | 0.017 | 0.009 |
| RBF SVM | 1000 | 0 | test_A | 0.117 | 0.618 | 0.023 | 0.014 |
| RBF SVM | 1000 | 0 | test_B | 0.155 | 0.646 | 0.017 | 0.009 |
| Gradient boosted trees | 1000 | 0 | test_A | 0.120 | 0.611 | 0.023 | 0.014 |
| Gradient boosted trees | 1000 | 0 | test_B | 0.138 | 0.643 | 0.018 | 0.009 |
| QSVM, ZZ entangling map | 1000 | 0 | test_A | 0.097 | 0.614 | 0.021 | 0.014 |
| QSVM, ZZ entangling map | 1000 | 0 | test_B | 0.125 | 0.641 | 0.017 | 0.009 |
| QSVM, product map (control) | 1000 | 0 | test_A | 0.111 | 0.621 | 0.022 | 0.014 |
| QSVM, product map (control) | 1000 | 0 | test_B | 0.124 | 0.642 | 0.016 | 0.009 |
| Reference: logistic regression, full pool | 471880 | -1 | test_A | 0.129 | 0.613 | 0.023 | 0.014 |
| Reference: logistic regression, full pool | 471880 | -1 | test_B | 0.150 | 0.611 | 0.017 | 0.009 |
| Reference: boosted trees, full pool | 471880 | -1 | test_A | 0.142 | 0.642 | 0.027 | 0.014 |
| Reference: boosted trees, full pool | 471880 | -1 | test_B | 0.138 | 0.633 | 0.016 | 0.009 |
<!-- END runs -->

### 7.1 Headline comparison at N = 2000

<!-- BEGIN headline -->
Not available: the results files for this block are missing (TypeError).
<!-- END headline -->

Paired differences (same bootstrap resamples):

<!-- BEGIN paired -->
Not available: the results files for this block are missing (TypeError).
<!-- END paired -->

![Headline intervals](results/headline_intervals.png)

### 7.2 Learning curves

![Learning curves](results/learning_curves.png)

<!-- BEGIN learning -->
| Test set | Model | N = 1000 |
|---|---|---|
| Internal test (set A) | Logistic regression | 0.127 ± 0.000 |
| Internal test (set A) | RBF SVM | 0.117 ± 0.000 |
| Internal test (set A) | Gradient boosted trees | 0.120 ± 0.000 |
| Internal test (set A) | QSVM, ZZ entangling map | 0.097 ± 0.000 |
| Internal test (set A) | QSVM, product map (control) | 0.111 ± 0.000 |
| External test (set B) | Logistic regression | 0.149 ± 0.000 |
| External test (set B) | RBF SVM | 0.155 ± 0.000 |
| External test (set B) | Gradient boosted trees | 0.138 ± 0.000 |
| External test (set B) | QSVM, ZZ entangling map | 0.125 ± 0.000 |
| External test (set B) | QSVM, product map (control) | 0.124 ± 0.000 |

Values are mean ± standard deviation over the draw seeds.
<!-- END learning -->

### 7.3 Deployment style operating points at N = 2000

Thresholds chosen on validation data, mean over the 3 draws.

<!-- BEGIN deployment -->
Not available: the results files for this block are missing (TypeError).
<!-- END deployment -->

### 7.4 Selected configurations at N = 2000

<!-- BEGIN params -->
Not available: the results files for this block are missing (TypeError).
<!-- END params -->

### 7.5 Quantum resources and efficiency

Circuit counts after transpiling to the basis {rz, sx, x, cx} with linear qubit connectivity (Qiskit `transpile`, optimisation level 1). The kernel circuit is the compute uncompute circuit U(x) followed by the inverse of U(y), with measurement.

<!-- BEGIN resources -->
| Map | Repetitions | Circuit | Qubits | Depth | Two qubit gates (CX) | Total gates |
|---|---|---|---|---|---|---|
| ZZ linear | 1 | feature map | 8 | 25 | 14 | 53 |
| ZZ linear | 1 | kernel circuit | 8 | 49 | 26 | 104 |
| ZZ linear | 2 | feature map | 8 | 35 | 28 | 106 |
| ZZ linear | 2 | kernel circuit | 8 | 69 | 54 | 210 |
| product | 1 | feature map | 8 | 4 | 0 | 32 |
| product | 1 | kernel circuit | 8 | 9 | 0 | 64 |
| product | 2 | feature map | 8 | 8 | 0 | 64 |
| product | 2 | kernel circuit | 8 | 17 | 0 | 128 |
<!-- END resources -->

Timings (mean over draws) and the number of distinct circuits each estimator would need on hardware. With exact states, one state per training and evaluation row suffices. A sampled pairwise estimator needs one circuit per training pair, N(N minus 1)/2, plus one per evaluation row and training row pair. All quantum timings are classical simulation times.

<!-- BEGIN efficiency -->
| Model | N | Tuning time (s) | Final fit time (s) | Prediction time per 1000 rows (ms) | Support vectors | Circuits, exact states (N plus evaluation rows) | Circuits, sampled pairwise estimator |
|---|---|---|---|---|---|---|---|
| Gradient boosted trees | 1000 | 12.5 | 0.09 | 1.0 |  |  |  |
| Logistic regression | 1000 | 0.3 | 0.01 | 0.0 |  |  |  |
| QSVM, product map (control) | 1000 | 1.9 | 0.08 | 48.1 | 907 | 423,024 | 422,523,500 |
| QSVM, ZZ entangling map | 1000 | 2.2 | 0.08 | 44.2 | 905 | 423,024 | 422,523,500 |
| RBF SVM | 1000 | 3.0 | 0.05 | 117.0 | 906 |  |  |

Evaluation rows per fitted model (validation plus both test subsets): 422,024.
<!-- END efficiency -->

### 7.6 Shot noise

At N = 500 (draw seed 0, the tuned entangling configuration), every kernel entry, training and test, was replaced by a binomial estimate with a finite number of shots. Noisy training Gram matrices that were not positive semi definite were projected by clipping negative eigenvalues to zero.

<!-- BEGIN shots -->
The shot noise study was not run.
<!-- END shots -->

![Shot noise](results/shot_noise.png)

## 8. Trade offs, advantages and limitations

Text in this section is written after the final run.

## 9. Reproducibility

Environment used: Windows 11, Python 3.13, 16 logical CPU cores, packages pinned in `requirements.txt` (Qiskit 2.6.1, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0). Seeds: split seed 2026, subset seed 2027, draw seeds 0, 1 and 2, bootstrap seed 2026, shot noise seed 0.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py  # about 40,000 small files, 20 to 30 minutes
bash scripts/run_all.sh          # tests, smoke run, full run, README tables
```

`bash scripts/run_all.sh --smoke` runs the tests and the smoke pipeline only (about 300 patients per set, a few minutes) and writes to `results/smoke/`, which is not committed. The full run writes incremental files to `results/` and can be resumed.

<!-- BEGIN runtime -->
Full run wall clock time: 6.1 minutes on the machine described above (sizes [1000], draw seeds [0], 500 bootstrap resamples).
<!-- END runtime -->

## 10. Requirement checklist

| Requirement | Where |
|---|---|
| Step 1: define the healthcare problem, relevance, data and intended outcome | Sections 2 and 3 |
| Step 2: strong classical baseline, credit, configuration and performance | Section 4, results in section 7 |
| Step 3: quantum contribution, why it is appropriate, which limitation it addresses | Section 5 |
| Step 4: fair evaluation with consistent data and conditions, results, improvements and limitations | Sections 6, 7 and 8 |
| Functional codebase | `src/sepsis6q/`, `scripts/`, `tests/` |
| Comparative evaluation with classical metrics and quantum resource and efficiency metrics | Sections 7.1 to 7.6 |
| Explanation of approach, advantages, limitations and trade offs | Sections 5 and 8 |
| Classical baselines comparable to the quantum approach | Sections 3.4 and 4 (same features, rows, preprocessing and tuning protocol) |

## 11. References

1. Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine 48(2), 2020. https://doi.org/10.1097/CCM.0000000000004145
2. PhysioNet/Computing in Cardiology Challenge 2019 dataset, version 1.0.0. https://doi.org/10.13026/v64v-d857 , https://physionet.org/content/challenge-2019/1.0.0/
3. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23), 2000.
4. Lashen H, et al. JMIR AI 2023;2:e45257 (in hospital deterioration prediction, Cleveland Clinic Abu Dhabi with NYU Abu Dhabi).
5. Havlicek V, Corcoles AD, Temme K, et al. Supervised learning with quantum enhanced feature spaces. Nature 567, 209 to 212, 2019.
6. Pedregosa F, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 2011.
7. Qiskit contributors. Qiskit: an open source framework for quantum computing. https://github.com/Qiskit/qiskit

## 12. Team

Team Kanban.

* Huda Mueen, team lead, Data Science and AI, Al Ain University, Abu Dhabi campus
* Awaiz Ahmed, Software Engineering, Al Ain University, Abu Dhabi campus
* Bilal Feroz, Software Engineering, Al Ain University, Abu Dhabi campus
* Kareena Mueenuddin, Mechatronics Engineering, Amity University
