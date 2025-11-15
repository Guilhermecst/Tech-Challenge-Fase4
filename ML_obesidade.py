# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import joblib
# %%
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# %%
df = pd.read_csv('data/Obesidade.csv')
# %%
df.head()
# %%
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders.items():
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy
# %%
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_mappings):
        self.feature_mappings = feature_mappings
        self.columns = list(feature_mappings.keys())
        self.encoder = None

    def fit(self, X, y=None):
        ordered_categories = [self.feature_mappings[col] for col in self.columns]
        
        self.encoder = OrdinalEncoder(categories=ordered_categories)
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        X_copy[self.columns] = self.encoder.transform(X_copy[self.columns])
        return X_copy
# %%
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode, handle_unknown='ignore'):
        self.columns_to_encode = columns_to_encode
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.new_feature_names = None

    def fit(self, X, y=None):
        cols_in_df = [col for col in self.columns_to_encode if col in X.columns]
        if not cols_in_df:
            return self
            
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown=self.handle_unknown)
        self.encoder.fit(X[cols_in_df])
        
        self.new_feature_names = self.encoder.get_feature_names_out(cols_in_df)
        
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        cols_in_df = [col for col in self.columns_to_encode if col in X.columns]
        if not cols_in_df or self.encoder is None:
            return X_copy
        
        one_hot_data = self.encoder.transform(X_copy[cols_in_df])
        
        df_onehot = pd.DataFrame(one_hot_data, columns=self.new_feature_names, index=X_copy.index)
        
        X_copy = X_copy.drop(columns=cols_in_df)
        
        return pd.concat([X_copy, df_onehot], axis=1)
# %%
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        cols_in_df = [col for col in self.columns_to_scale if col in X_copy.columns]
        if not cols_in_df:
            return X_copy

        X_copy[cols_in_df] = self.scaler.transform(X_copy[cols_in_df])
        return X_copy
# %%
class IMCCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        X_copy['IMC'] = X_copy['Peso'] / (X_copy['Altura'] ** 2)
        
        return X_copy
# %%
features_to_use = [col for col in df.columns if col not in ['Nivel_obesidade', 'IMC']]
X = df[features_to_use]
y = df['Nivel_obesidade']
# %%
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
joblib.dump(target_encoder, 'target_encoder_obesidade.joblib')
# %%
binary_features = ['Sexo_biologico', 'Historico_familiar_excesso_peso', 'Consumo_frequente_alimentos_caloricos', 'Habito_fumar', 'Monitoramento_ingestao_calorica']
ordinal_features = ['Consumo_lanches_entre_refeicoes', 'Consumo_bebida_alcoolica']
nominal_features = ['Meio_transporte_habitual']
numerical_features = ['Idade', 'Altura', 'Peso', 'Frequencia_consumo_vegetais', 'Numero_refeicoes_principais', 'Consumo_diario_agua', 'Frequencia_atividade_fisica_semanal', 'Tempo_diario_dispositivos_eletronicos']
# %%
ordinal_mappings = [
    ['Não', 'As_vezes', 'Frequentemente', 'Sempre'], # 'Consumo_lanches_entre_refeicoes'
    ['Não', 'As_vezes', 'Frequentemente', 'Sempre']  # 'Consumo_bebida_alcoolica'
]
# %%
preprocessor = ColumnTransformer(
    transformers=[
        # Para features binárias (Sim/Não) e nominais (Masculino/Feminino), OneHotEncoder é o mais seguro
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), binary_features + nominal_features),
        
        # Para features ordinais com uma ordem clara
        ('ordinal', OrdinalEncoder(categories=ordinal_mappings), ordinal_features),
        
        # Para todas as features numéricas
        ('numeric', StandardScaler(), numerical_features)
    ],
    remainder='passthrough' # Mantém colunas não especificadas, se houver
)
# %%
model_pipeline_knn = Pipeline(steps=[
    ('imc_calculator', IMCCalculator()),
    ('preprocessor', preprocessor),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', KNeighborsClassifier(
        n_neighbors=1))
])
# %%
model_pipeline_rf = Pipeline(steps=[
    ('imc_calculator', IMCCalculator()),
    ('preprocessor', preprocessor),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    ))
])
# %%
model_pipeline_svc = Pipeline(steps=[
    ('imc_calculator', IMCCalculator()),
    ('preprocessor', preprocessor),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', SVC(
        random_state=42
    ))
])
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)
# %%
class_names = target_encoder.classes_
# %%
model_pipeline_knn.fit(X_train, y_train)
# %%
y_pred = model_pipeline_knn.predict(X_test)
# %%
print("--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=class_names))
# %%
print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# %%
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.show()
# %%
# %%
model_pipeline_rf.fit(X_train, y_train)
# %%
y_pred = model_pipeline_rf.predict(X_test)
# %%
print("--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=class_names))
# %%
print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# %%
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.show()
# %%
model_pipeline_svc.fit(X_train, y_train)
# %%
y_pred = model_pipeline_svc.predict(X_test)
# %%
print("--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=class_names))
# %%
print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# %%
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.show()
# %%
joblib.dump(model_pipeline_svc, 'modelo_svc.joblib')
# %%
