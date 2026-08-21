# 🧮 Obesity Level Calculator

## 🧠 Project Description
The **Obesity Level Calculator** project uses **Machine Learning** techniques to predict a person's obesity level based on physical and behavioral information, such as age, eating habits, family history, and physical activity level.

The solution was deployed as an interactive web interface built with **Streamlit**, allowing users to enter their data and receive an automatic prediction.

🔗 **Access the app:** [calculadora-nivel-obesidade-tc-4-fiap.streamlit.app](https://calculadora-nivel-obesidade-tc-4-fiap.streamlit.app/)

---

## 📊 Dataset
The dataset used is **Obesidade.csv**, containing physical profile and lifestyle attributes. Each record represents a person with their corresponding classified obesity level.

### Main variables:
- `Idade` (Age), `Altura` (Height), `Peso` (Weight)
- `Sexo_biologico` (Biological sex)
- `Consumo_frequente_alimentos_caloricos` (Frequent consumption of caloric food)
- `Frequencia_atividade_fisica_semanal` (Weekly physical activity frequency)
- `Consumo_diario_agua` (Daily water intake)
- `Numero_refeicoes_principais` (Number of main meals)
- `Tempo_diario_dispositivos_eletronicos` (Daily time on electronic devices)
- `Meio_transporte_habitual` (Usual mode of transportation)
- `Nivel_obesidade` (Obesity level) *(target variable)*

---

## ⚙️ Machine Learning Pipeline Structure

The project implements a **complete pipeline** using custom classes and components from Scikit-learn and Imbalanced-learn.

### Pipeline steps:
1. **BMI Calculation**
   Adds a new feature `BMI = Weight / Height²` through the custom `IMCCalculator` class.

2. **Data preprocessing**
   Applied via `ColumnTransformer`:
   - `OneHotEncoder` → binary and nominal variables.
   - `OrdinalEncoder` → ordinal variables with hierarchical order.
   - `StandardScaler` → normalization of numeric variables.

3. **Class balancing**
   Uses **SMOTE (Synthetic Minority Oversampling Technique)** to handle data imbalance.

4. **Training and Evaluation**
   Models tested:
   - `KNeighborsClassifier`
   - `RandomForestClassifier`
   - `SVC (Support Vector Machine)`

   The **SVC** model showed the best performance and was selected for the final application.

---

## 🧩 Folder and File Structure

```
📂 projeto_obesidade/
│
├── data/
│   └── Obesidade.csv                     # Dataset
│
├── modelo_svc.joblib                     # Trained model
├── target_encoder_obesidade.joblib       # Target variable encoder
│
├── app.py                                # Streamlit application
├── treino_modelos.py                     # Training and evaluation script
│
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🧰 Technologies Used
- **Python 3.10+**
- **Pandas** – Data manipulation and analysis
- **Scikit-learn** – Preprocessing and modeling
- **Imbalanced-learn** – Class balancing (SMOTE)
- **Matplotlib** – Results visualization
- **Joblib** – Trained model persistence
- **Streamlit** – Interactive web interface

---

## 📈 How to Run the Project

### 1️⃣ Clone the repository
```
git clone https://github.com/<usuario>/calculadora-obesidade.git
cd calculadora-obesidade
```

### 2️⃣ Create the virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate  # (on Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 3️⃣ Run the training script
```
python treino_modelos.py
```

### 4️⃣ Run the Streamlit application
```
streamlit run app.py
```

---

## 🔍 Results and Evaluation

The models were evaluated using **classification metrics (precision, recall, F1-score)** and a **confusion matrix**.

The **SVC** model showed the best balance between precision and generalization, and was saved as `modelo_svc.joblib` for real-time predictions in the application.

---

## 🚀 Deployment

The application is publicly available via **Streamlit Cloud**, allowing users to interact directly with the trained model.

---

## 👨‍💻 Author
**Guilherme Costa**
🧩 Data Analyst/Scientist
📧 [guilherme.cst@outlook.com.br]
🔗 [linkedin.com/in/silva-guilherme-costa](https://www.linkedin.com/in/silva-guilherme-costa/)

---
