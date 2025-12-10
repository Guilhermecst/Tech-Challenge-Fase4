import streamlit as st
import pandas as pd
import joblib
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- CLASSES CUSTOMIZADAS (OBRIGATÓRIAS PARA CARREGAR O MODELO) ---
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


class IMCCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy['IMC'] = X_copy['Peso'] / (X_copy['Altura'] ** 2)
        return X_copy


# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(
    page_title="Preditor de Nível de Obesidade",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOMIZADO ---
st.markdown("""
<style>
.st-emotion-cache-1r6slb0 {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 25px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    height: 100%;
}
h1 { color: #333333; }
h3 { color: #1f77b4; font-weight: bold; }
div.stButton > button {
    background-color: #28a745;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover {
    background-color: #218838;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
div[data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# --- MAPA DE VALORES PARA A INTERFACE ---
map_transporte = {
    'Transporte publico': 'Transporte_publico',
    'Carro': 'Carro',
    'A pe': 'A_pe',
    'Motocicleta': 'Motocicleta',
    'Bicicleta': 'Bicicleta'
}
map_lanches_alcool = {
    'Não': 'Não',
    'As vezes': 'As_vezes',
    'Frequentemente': 'Frequentemente',
    'Sempre': 'Sempre'
}

# --- BARRA LATERAL ---
with st.sidebar:
    st.title('Calculadora de Obesidade')
    st.markdown("Este aplicativo usa Machine Learning para prever o nível de obesidade.")
    st.markdown("---")
    st.subheader("Status do Modelo")

    MODEL_LOADED = False
    pipeline = None
    target_encoder = None

    try:
        # >>> AQUI: CARREGANDO O MODELO KNN <<<
        pipeline = joblib.load('modelo_knn.joblib')
        target_encoder = joblib.load('target_encoder_obesidade.joblib')
        st.success("✅ Modelo KNN carregado com sucesso!")
        MODEL_LOADED = True
    except FileNotFoundError:
        st.error("❌ Arquivos do modelo não encontrados (modelo_knn.joblib ou target_encoder_obesidade.joblib).")
        st.warning("Execute o script de treinamento primeiro e verifique os caminhos.")
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivos: {str(e)}")

    st.markdown("---")
    st.info("Projeto de análise de dados em saúde.")

tab1, tab2 = st.tabs(["🔮 Predição", "📊 Dashboards"])

# --- INTERFACE PRINCIPAL ---
with tab1:
    st.title('🩺 Preencha os campos para a previsão')

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container():
            st.subheader("👤 Informações Pessoais e Histórico")
            sexo = st.selectbox('Sexo biológico', ['Masculino', 'Feminino'])
            idade = st.number_input('Idade', min_value=1, max_value=100, value=30)
            altura = st.number_input('Altura (m)', min_value=1.0, max_value=2.5, value=1.70, format="%.2f")
            peso = st.number_input('Peso (kg)', min_value=30.0, max_value=250.0, value=70.0, format="%.1f")
            hist_familiar = st.radio('Histórico familiar de sobrepeso?', ['Sim', 'Não'], horizontal=True)
            fumo = st.radio('Você fuma?', ['Sim', 'Não'], horizontal=True)

    with col2:
        with st.container():
            st.subheader("🥗 Alimentação")
            cons_alim_caloricos = st.radio('Consumo frequente de alimentos calóricos?', ['Sim', 'Não'], horizontal=True)
            freq_cons_veg = st.slider('Frequência de consumo de vegetais', 1, 3, 2,
                                      help="1: Nunca, 2: Às vezes, 3: Sempre")
            num_refeicoes = st.slider('Número de refeições principais diárias', 1, 5, 3)
            cons_lanches_display = st.select_slider(
                'Consumo de lanches entre refeições',
                options=list(map_lanches_alcool.keys()),
                value='As vezes'
            )
            cons_agua = st.slider('Consumo diário de água (Litros)', 1, 4, 2)
            cons_alcool_display = st.select_slider(
                'Consumo de bebida alcoólica',
                options=list(map_lanches_alcool.keys()),
                value='Não'
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.container():
        st.subheader("🏃 Rotina e Transporte")
        bottom_col1, bottom_col2, bottom_col3 = st.columns(3, gap="large")
        with bottom_col1:
            monitor_calorias = st.radio('Faz monitoramento de calorias?', ['Sim', 'Não'], horizontal=True)
            freq_ativ_fisica = st.slider('Atividade física (dias/semana)', 0, 7, 2)
        with bottom_col2:
            tempo_telas = st.slider('Tempo diário em telas (horas)', 0.0, 10.0, 2.0, step=0.5)
        with bottom_col3:
            meio_transporte_display = st.selectbox(
                'Meio de transporte habitual',
                options=list(map_transporte.keys())
            )

    # --- BOTÃO DE PREVISÃO E RESULTADO ---
    if st.button('**✨ Calcular Nível de Obesidade**', use_container_width=True, type="primary"):
        if not MODEL_LOADED:
            st.error("❌ O modelo não está carregado. Não é possível fazer a previsão.")
        else:
            # Converte os valores da interface para o formato do modelo
            cons_lanches_modelo = map_lanches_alcool[cons_lanches_display]
            cons_alcool_modelo = map_lanches_alcool[cons_alcool_display]
            meio_transporte_modelo = map_transporte[meio_transporte_display]

            dados_usuario = pd.DataFrame({
                'Sexo_biologico': [sexo],
                'Idade': [idade],
                'Altura': [altura],
                'Peso': [peso],
                'Historico_familiar_excesso_peso': [hist_familiar],
                'Consumo_frequente_alimentos_caloricos': [cons_alim_caloricos],
                'Frequencia_consumo_vegetais': [float(freq_cons_veg)],
                'Numero_refeicoes_principais': [float(num_refeicoes)],
                'Consumo_lanches_entre_refeicoes': [cons_lanches_modelo],
                'Habito_fumar': [fumo],
                'Consumo_diario_agua': [float(cons_agua)],
                'Monitoramento_ingestao_calorica': [monitor_calorias],
                'Frequencia_atividade_fisica_semanal': [float(freq_ativ_fisica)],
                'Tempo_diario_dispositivos_eletronicos': [float(tempo_telas)],
                'Consumo_bebida_alcoolica': [cons_alcool_modelo],
                'Meio_transporte_habitual': [meio_transporte_modelo]
            })

            try:
                predicao_codificada = pipeline.predict(dados_usuario)
                resultado_com_underline = target_encoder.inverse_transform(predicao_codificada)
                resultado_final = resultado_com_underline[0].replace('_', ' ')
                imc_calculado = peso / (altura ** 2)

                st.markdown("---")
                st.header("📊 Resultados da Análise")

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="IMC Calculado", value=f"{imc_calculado:.2f}")
                with res_col2:
                    st.metric(label="Previsão do Nível de Obesidade", value=resultado_final)

                st.success(f"✅ Previsão: **{resultado_final}**")

            except Exception as e:
                st.error(f"❌ Erro durante a predição: {str(e)}")
                st.warning("Verifique se os dados inseridos são consistentes.")

# --- ABA 2: DASHBOARDS ---
with tab2:
    st.header("📊 Dashboards e Visualizações")

    try:
        df_obesidade = pd.read_csv('data/Obesidade.csv')
        st.success(f"✅ Dataset carregado: {len(df_obesidade):,} registros")
    except FileNotFoundError:
        st.error("❌ Arquivo 'data/Obesidade.csv' não encontrado!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()

    mulheres = df_obesidade[df_obesidade['Sexo_biologico'] == 'Feminino']
    homens = df_obesidade[df_obesidade['Sexo_biologico'] == 'Masculino']

    ordem = [
        'Abaixo_do_peso', 'Peso_normal', 'Sobrepeso_nivel_I',
        'Sobrepeso_nivel_II', 'Obesidade_tipo_I',
        'Obesidade_tipo_II', 'Obesidade_tipo_III'
    ]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", f"{len(df_obesidade):,}")
    with col2:
        st.metric("Mulheres", f"{len(mulheres):,}")
    with col3:
        st.metric("Homens", f"{len(homens):,}")
    with col4:
        st.metric("IMC Médio", f"{df_obesidade['IMC'].mean():.1f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Distribuição de Peso")
        fig_peso, ax_peso = plt.subplots(figsize=(8, 5))
        sns.histplot(df_obesidade['Peso'], bins=25, kde=True, color='forestgreen', alpha=0.6, ax=ax_peso)
        ax_peso.axvline(df_obesidade['Peso'].mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Média: {df_obesidade["Peso"].mean():.1f}kg')
        ax_peso.set_title('Distribuição de Peso (Geral)')
        ax_peso.legend()
        ax_peso.grid(True, alpha=0.3)
        st.pyplot(fig_peso)

    with col2:
        st.subheader("📈 Distribuição de IMC")
        fig_imc, ax_imc = plt.subplots(figsize=(8, 5))
        sns.histplot(df_obesidade['IMC'], bins=25, kde=True, color='orange', alpha=0.6, ax=ax_imc)
        ax_imc.axvline(df_obesidade['IMC'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Média: {df_obesidade["IMC"].mean():.1f}')
        ax_imc.set_title('Distribuição de IMC (Geral)')
        ax_imc.legend()
        ax_imc.grid(True, alpha=0.3)
        st.pyplot(fig_imc)

    st.subheader("👥 Comparação por Sexo")

    fig_sexo_peso, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(mulheres['Peso'], bins=20, kde=True, color='hotpink', alpha=0.7, ax=ax1)
    ax1.set_title('Mulheres')
    ax1.axvline(mulheres['Peso'].mean(), color='red', linestyle='--',
                label=f'Média: {mulheres["Peso"].mean():.1f}')
    ax1.legend()

    sns.histplot(homens['Peso'], bins=20, kde=True, color='royalblue', alpha=0.7, ax=ax2)
    ax2.set_title('Homens')
    ax2.axvline(homens['Peso'].mean(), color='red', linestyle='--',
                label=f'Média: {homens["Peso"].mean():.1f}')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig_sexo_peso)

    st.subheader("📊 Níveis de Obesidade por Grupo")

    freq_geral = df_obesidade['Nivel_obesidade'].value_counts().reindex(ordem, fill_value=0)
    freq_mulheres = mulheres['Nivel_obesidade'].value_counts().reindex(ordem, fill_value=0)
    freq_homens = homens['Nivel_obesidade'].value_counts().reindex(ordem, fill_value=0)

    x = np.arange(len(ordem))
    largura = 0.25

    fig_barras, ax_barras = plt.subplots(figsize=(14, 7))
    bars0 = ax_barras.bar(x - largura, freq_geral, largura, label='Geral', color='#90c490', edgecolor='black', alpha=0.8)
    bars1 = ax_barras.bar(x, freq_mulheres, largura, label='Mulheres', color='#ffb3d9', edgecolor='black', alpha=0.8)
    bars2 = ax_barras.bar(x + largura, freq_homens, largura, label='Homens', color='#9fb3ef', edgecolor='black', alpha=0.8)

    ax_barras.set_ylabel('Frequência')
    ax_barras.set_title('Distribuição dos Níveis de Obesidade')
    ax_barras.set_xticks(x)
    ax_barras.set_xticklabels([label.replace('_', ' ').title() for label in ordem],
                              rotation=45, ha='right')
    ax_barras.legend()
    ax_barras.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars0 + bars1 + bars2:
        height = bar.get_height()
        ax_barras.text(bar.get_x() + bar.get_width() / 2., height + 5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_barras)

    st.subheader("🔍 Insights Importantes")

    col1, col2, col3 = st.columns(3)
    with col1:
        perc_sobrepeso_obesidade = len(df_obesidade[df_obesidade['Nivel_obesidade'].isin([
            'Sobrepeso_nivel_I', 'Sobrepeso_nivel_II', 'Obesidade_tipo_I',
            'Obesidade_tipo_II', 'Obesidade_tipo_III'])]) / len(df_obesidade) * 100
        st.metric("Sobrepeso + Obesidade", f"{perc_sobrepeso_obesidade:.1f}%")

    with col2:
        with_hist = len(df_obesidade[df_obesidade['Historico_familiar_excesso_peso'] == 'Sim'])
        st.metric("Histórico Familiar", f"{with_hist:,} pessoas")

    with col3:
        media_idade_obesos = df_obesidade[df_obesidade['Nivel_obesidade'].str.contains('Obesidade')]['Idade'].mean()
        st.metric("Idade Média Obesos", f"{media_idade_obesos:.0f} anos")
