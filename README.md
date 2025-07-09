# 🤖 Diabetes Prediction with Artificial Neural Networks

This project uses **Multilayer Perceptron Neural Networks (MLPClassifier)** to predict the presence of **diabetes** based on clinical and demographic data. It also applies **data balancing (SMOTE)** and **normalization** techniques.

---

## 📊 Dataset

* Dataset: Clinical data from patients with or without diabetes
* Source: Simulated dataset with over 1300 records
* Features used:

  * `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`
  * Target: `diabetes` (0 = no, 1 = yes)

---

## ⚙️ Technologies

* Python 3.x
* Jupyter Notebook
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`

---

## 🧠 What Was Done

* Data distribution analysis and missing value detection
* Transformation of categorical variables into numerical format
* Normalization using `MinMaxScaler`
* Minority class balancing with **SMOTE**
* Training of a neural network with:

  * 2 hidden layers with 490 neurons each
  * ReLU activation, Adam optimizer
  * L2 regularization (`alpha=1e-4`)
  * Up to 1000 iterations

---

## 📈 Results

* **Model accuracy**: \~96%
* **Precision for non-diabetic cases**: 98%
* **Precision for diabetic cases**: 89%
* Most relevant features: `HbA1c_level`, `blood_glucose_level`, `bmi`, `age`, `hypertension`

---

## 🔬 Validation

Tests were performed with simulated user profiles:

* The model correctly identified patients with multiple risk factors
* For healthy profiles, the model returned 0% probability of diabetes
* In borderline cases or isolated risk factors (e.g., high HbA1c only), the model underestimated the risk

---

## 📁 How to Run

1. Clone the repository:

```bash
git clone https://github.com/ViniciusCastellani/diabetes-prediction-neuralnetwork
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

3. Run the notebook:

```bash
jupyter notebook exercicio_rede_neural_diabetes_COM_SMOTE.ipynb
```