
# 🛡️ Fraud Detection Predictor App

**Can a simple app predict fraud like a financial analyst?**  
This project is a full pipeline from raw Kaggle dataset → exploratory data analysis → logistic regression pipeline → live Streamlit app.

> Built for education, intuition, and real-world simulation. You can inspect, interact, and even deploy it.

---

<p align="center">
  <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" height="45" />
</p>

---

## 📦 Dataset Used

**Fraud Detection Dataset by Aman Ali Siddiqui**  
🔗 [Kaggle Source](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)

> A synthetic dataset of over 6.3 million financial transactions between customers and merchants.

### Key Columns

| Column Name      | Description                                          |
|------------------|------------------------------------------------------|
| `step`           | Time step (hourly)                                   |
| `type`           | Transaction type (PAYMENT, CASH_OUT, TRANSFER, etc.) |
| `amount`         | Amount of transaction                                |
| `oldbalanceOrg`  | Sender's balance before transaction                  |
| `newbalanceOrig` | Sender's balance after transaction                   |
| `oldbalanceDest` | Receiver's balance before transaction                |
| `newbalanceDest` | Receiver's balance after transaction                 |
| `isFraud`        | Target label (1 = fraud, 0 = legit)                  |

---

## 🧪 Exploratory Data Analysis

This dataset is **imbalanced**, with only ~0.13% frauds. But a careful look reveals patterns:

---

### 🔢 Transaction Types Distribution

<p align="center"><img src="https://github.com/sparshb4tra/Fraud_Detection/blob/main/plots/Transaction%20Types.png" width="400"></p>

> Most transactions are `CASH_OUT`, `PAYMENT`, and `CASH_IN`.

---

### 🚩 Fraud Rate by Transaction Type

<p align="center"><img src="https://github.com/sparshb4tra/Fraud_Detection/blob/main/plots/Fraud%20Rate%20by%20Type.png" width="400"></p>

> 🔥 **Only `TRANSFER` and `CASH_OUT` show fraud**. This insight powers our feature filtering and Streamlit form.

---

### 💰 Amount Distribution (Log Scale)

<p align="center"><img src="https://github.com/sparshb4tra/Fraud_Detection/blob/main/plots/Transaction%20Amount%20Distribution%20(log%20scale).png" width="400"></p>

> The raw `amount` is skewed, so we log-transformed it using `np.log1p` before modeling.

---

### 📦 Boxplot: Fraud vs Amount (Under ₹50000)

<p align="center"><img src="https://github.com/sparshb4tra/Fraud_Detection/blob/main/plots/Amount%20vs%20ifFraud%20(Filtered%20under%2050k).png" width="400"></p>

> Even small transactions can be fraudulent — often overlooked.

---

### 🕒 Fraud Timing Pattern

<p align="center"><img src="https://github.com/sparshb4tra/Fraud_Detection/blob/main/plots/Frauds%20Over%20Time.png" width="450"></p>

> Fraud spikes at specific hours. Attackers may exploit system lulls.

---

### 🔍 Breakdown by Transaction Type

<p align="center"><img src="https://github.com/sparshb4tra/Fraud_Detection/blob/main/plots/Fraud%20Distribution%20in%20Transfer%20%26%20Cash_Out.png" width="450"></p>

> Most frauds happen in `TRANSFER`, followed by `CASH_OUT`. Others are consistently safe.

---

### 🧊 Correlation Matrix

<p align="center"><img src="https://github.com/sparshb4tra/Fraud_Detection/blob/main/plots/Correlation%20Matrix.png" width="450"></p>

> High positive correlation between:
- `oldbalanceOrg` ↔ `newbalanceOrig`
- `amount` is moderately negative with remaining balances

---

## ⚙️ Feature Engineering

We added:
- `balanceDiffOrig` = how much the sender lost
- `balanceDiffDest` = how much the receiver gained

And dropped:
- `step`, `nameOrig`, `nameDest`, and `isFlaggedFraud`

---

## 🧠 Model Building

| Step            | Method/Tool                    |
|-----------------|--------------------------------|
| Preprocessing   | `ColumnTransformer` with OneHotEncoder + StandardScaler |
| Classifier      | `LogisticRegression` with `class_weight="balanced"`    |
| Evaluation      | `confusion_matrix`, `classification_report`            |
| Deployment      | `joblib.dump()` of full pipeline                        |

```python
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])
````

> Accuracy: \~99.96%
> Recall: Much better than naive classifiers due to class balancing

---

## 🖥️ Streamlit App

The app provides:

* A minimal and elegant UI
* Fields to enter a transaction manually
* Real-time fraud prediction using the trained pipeline


---

## 🚀 Run Locally

```bash
# Clone the project
git clone https://github.com/yourusername/fraud-detection-app.git
cd fraud-detection-app

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run fraud_detection.py
```

---

## 🌐 Deploy on Streamlit Cloud

The easiest way:

1. Push code to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repo, select `fraud_detection.py`
4. Done!

---

## 📁 Project Structure

```
fraud-detection-app/
├── analysis_model.ipynb          # Full EDA + model training notebook
├── fraud_detection.py            # Streamlit app frontend
├── fraud_detection_pipeline.pkl  # Trained model pipeline
├── AIML Dataset.csv              # Raw Kaggle dataset
├── assets/                       # All images used in README
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🧾 Tech Stack

* Python 3.11
* Pandas, NumPy
* Seaborn, Matplotlib
* Scikit-learn
* Streamlit
* Joblib

---

## 🔮 Possible Next Steps

* ✅ Add `predict_proba()` to show model confidence
* 📤 Allow CSV upload for batch fraud detection
* 📥 Enable PDF report generation
* 🧩 Integrate with a Flask backend for REST API deployment

---

## 🙋‍♂️ Author

Made with ❤️ by [sparsh](https://github.com/sparshb4tra)
Inspired by the intersection of finance, trust, and data.

> “Fraud is smart. Your model needs to be smarter.”



