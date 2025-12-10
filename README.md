# 🧮 Calculadora de Nível de Obesidade  

## 🧠 Descrição do Projeto  
O projeto **Calculadora de Nível de Obesidade** utiliza técnicas de **Machine Learning** para prever o nível de obesidade de um indivíduo com base em informações físicas e comportamentais, como idade, hábitos alimentares, histórico familiar e nível de atividade física.  

A solução foi disponibilizada em uma interface web interativa desenvolvida com **Streamlit**, permitindo que usuários insiram seus dados e recebam uma previsão automática.  

🔗 **Acesse a aplicação:** [calculadora-nivel-obesidade-tc-4-fiap.streamlit.app](https://calculadora-nivel-obesidade-tc-4-fiap.streamlit.app/)

---

## 📊 Conjunto de Dados  
O dataset utilizado é o **Obesidade.csv**, contendo atributos de perfil físico e hábitos de vida. Cada registro representa uma pessoa com seu respectivo nível de obesidade classificado.  

### Principais variáveis:
- `Idade`, `Altura`, `Peso`  
- `Sexo_biologico`  
- `Consumo_frequente_alimentos_caloricos`  
- `Frequencia_atividade_fisica_semanal`  
- `Consumo_diario_agua`  
- `Numero_refeicoes_principais`  
- `Tempo_diario_dispositivos_eletronicos`  
- `Meio_transporte_habitual`  
- `Nivel_obesidade` *(variável alvo)*  

---

## ⚙️ Estrutura do Pipeline de Machine Learning  

O projeto implementa um **pipeline completo** de aprendizado de máquina utilizando classes personalizadas e componentes do Scikit-learn e Imbalanced-learn.  

### Etapas do pipeline:
1. **Cálculo do IMC (Índice de Massa Corporal)**  
   Adiciona uma nova feature `IMC = Peso / Altura²` através da classe customizada `IMCCalculator`.

2. **Pré-processamento de dados**  
   Aplicado via `ColumnTransformer`:
   - `OneHotEncoder` → variáveis binárias e nominais.  
   - `OrdinalEncoder` → variáveis ordinais com ordem hierárquica.  
   - `StandardScaler` → normalização de variáveis numéricas.

3. **Balanceamento de classes**  
   Utiliza **SMOTE (Synthetic Minority Oversampling Technique)** para lidar com desbalanceamento dos dados.

4. **Treinamento e Avaliação**  
   Modelos testados:
   - `KNeighborsClassifier`  
   - `RandomForestClassifier`  
   - `SVC (Support Vector Machine)`  

   O modelo **SVC** apresentou o melhor desempenho e foi selecionado para a aplicação final.

---

## 🧩 Estrutura de Pastas e Arquivos  

```
📂 projeto_obesidade/
│
├── data/
│   └── Obesidade.csv                     # Base de dados
│
├── modelo_svc.joblib                     # Modelo treinado
├── target_encoder_obesidade.joblib       # Encoder da variável alvo
│
├── app.py                                # Aplicação Streamlit
├── treino_modelos.py                     # Script de treino e avaliação
│
├── requirements.txt                      # Dependências do projeto
└── README.md                             # Documentação
```

---

## 🧰 Tecnologias Utilizadas  
- **Python 3.10+**  
- **Pandas** – Manipulação e análise de dados  
- **Scikit-learn** – Pré-processamento e modelagem  
- **Imbalanced-learn** – Balanceamento de classes (SMOTE)  
- **Matplotlib** – Visualização dos resultados  
- **Joblib** – Salvamento do modelo treinado  
- **Streamlit** – Interface web interativa  

---

## 📈 Como Executar o Projeto  

### 1️⃣ Clonar o repositório  
```
git clone https://github.com/<usuario>/calculadora-obesidade.git
cd calculadora-obesidade
```

### 2️⃣ Criar o ambiente virtual e instalar dependências  
```
python -m venv venv
source venv/bin/activate  # (no Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 3️⃣ Executar o script de treinamento  
```
python treino_modelos.py
```

### 4️⃣ Rodar a aplicação Streamlit  
```
streamlit run app.py
```

---

## 🔍 Resultados e Avaliação  

Os modelos foram avaliados com **métricas de classificação (precision, recall, F1-score)** e **matriz de confusão**.  

O modelo **SVC** apresentou o melhor equilíbrio entre precisão e generalização, sendo salvo como `modelo_svc.joblib` e utilizado pela aplicação para previsões em tempo real.

---

## 🚀 Implantação  

A aplicação está disponível publicamente via **Streamlit Cloud**, permitindo interação direta do usuário com o modelo treinado.  

---

## 👨‍💻 Autor  
**Guilherme Costa**  
🧩 Data Analyst/Scientist
📧 [guilherme.cst@outlook.com.br]  
🔗 [linkedin.com/in/seu-perfil](https://www.linkedin.com/in/silva-guilherme-costa/)

---

```