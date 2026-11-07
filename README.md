# 🩺 Chronic Kidney Disease (CKD) Diagnostic Benchmarking

An enterprise-grade, end-to-end Machine Learning pipeline developed to accurately classify whether a patient has **Chronic Kidney Disease (CKD)** or **Not** based on non-invasive clinical biometric features (e.g., Packed Cell Volume, Serum Creatinine, Blood Urea, and Age) using the official Kaggle dataset.

---

## 🚀 Key Engineering Enhancements

* **Zero Data Leakage Workflow:** Unlike naive setups where imputation or scaling is applied to the entire dataset simultaneously, this pipeline executes a `stratified train-test split` first. Advanced `KNNImputer` and `StandardScaler` transformations learn rules solely from the training partition to protect the evaluation subset from optimistic bias.
* **Dual-Metric Clinical Validation:** Evaluates models based on both **Accuracy** and **Recall (Sensitivity)**. In predictive healthcare applications, missing a true-positive CKD case carries massive patient risk; optimizing for Recall ensures the system reliably captures high-risk diagnoses.
* **Explainable AI (XAI):** Uses tree-based relative importance scoring to map exactly which clinical diagnostic indicators dictate the model's diagnostic assertions.

---

## 📊 Performance Benchmarking Results

After sanitizing the dataset, dropping structural identifiers, handling high-dimensional missing records, and isolating test data, the deployed machine learning architectures achieved the following test set outcomes:

| Model Architecture | Test Accuracy | Clinical Recall (Sensitivity) | Status |
| :--- | :---: | :---: | :---: |
| **Random Forest Ensemble** | **85.38%** | **95.35%** | **🏆 Recommended for Deployment** |
| **K-Nearest Neighbors (KNN)** | 85.00% | 93.02% | Baseline Competitor |
| **Logistic Regression Solver** | 84.62% | 91.28% | Parametric Baseline |

---

## 🛠️ Pipeline Architecture & Implementation

### 1. Data Cleaning & Unified Mapping
Standardizes noisy, manually logged categorical metrics (such as explicit strip routines for string spacing anomalies, structural regex replacement of missing text placeholders `?`, `\t?`, `nan` to operational `np.nan`), converts object flags to strict operational datatypes, and merges independent target variables into a unified structure.

### 2. Robust Feature Transformation
Prevents analytical instability by implementing instance-based multi-dimensional missing value imputation via `KNNImputer` alongside localized variance normalization via `StandardScaler`. 

### 3. Model Architecture Training
Trains a parametric generalized linear baseline algorithm (Logistic Regression) alongside instance-based cluster sorting (KNN) and an independent 100-tree voting framework (Random Forest Ensemble) to capture non-linear combinations across multi-dimensional feature environments.

### 4. Advanced Clinical Visualizations
* **Horizontal Benchmark Plots:** Synthesizes isolated accuracy scores to trace mathematical model variance.
* **Confusion Matrix Display:** Maps precise true vs. false outcomes to visually distinguish error rates.
* **Clinical Feature Importance:** Automatically ranks operational predictors, revealing that clinical indicators like `packed_cell_volume (PCV)` play a dominant role in final diagnostic weights.

---

## 💻 Tech Stack & Dependencies

* Python 3.x
* Pandas & NumPy (Data Wrangling & Extraction)
* Scikit-Learn (Pipelines, Estimators, Pre-processing, Analytics)
* Matplotlib & Seaborn (Medical Visualizations & Graph Plotting)

---

## 🔮 Future Deployments
The backend architecture is structured to fit seamlessly into containerized edge-microservices or operational web components (e.g., Streamlit frameworks), providing medical professionals with a reliable clinical decision-support tool.
