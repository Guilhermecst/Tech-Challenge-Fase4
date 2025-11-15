import streamlit as st
import pandas as pd
import joblib
import warnings
from ML_obesidade import IMCCalculator
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(
    page_title="Preditor de Nível de Obesidade",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CARREGAMENTO DO MODELO E ENCODERS ---
try:
    pipeline = joblib.load('modelo_svc.joblib')
    target_encoder = joblib.load('target_encoder_obesidade.joblib')
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("Arquivo do modelo ('modelo_svc.joblib') ou do encoder ('target_encoder_obesidade.joblib') não encontrado.")
    st.warning("Por favor, treine e salve seu modelo e encoder primeiro.")
    MODEL_LOADED = False
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar os arquivos: {e}")
    MODEL_LOADED = False

# --- INTERFACE DO USUÁRIO (Inputs) ---
st.title('Calculadora de Nível de Obesidade 🩺')
st.markdown("Preencha as informações abaixo para que o modelo de Machine Learning possa prever o nível de obesidade.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informações Pessoais")
    sexo = st.selectbox('Sexo biológico', ['Masculino', 'Feminino'])
    idade = st.number_input('Idade', min_value=1, max_value=100, value=30)
    altura = st.number_input('Altura (em metros)', min_value=1.0, max_value=2.5, value=1.70, format="%.2f")
    peso = st.number_input('Peso (em kg)', min_value=30.0, max_value=250.0, value=70.0, format="%.1f")
    
    st.subheader("Histórico e Hábitos")
    hist_familiar = st.radio(
        'Histórico familiar de excesso de peso?',
        ['Sim', 'Não'], horizontal=True
    )
    fumo = st.radio('Você fuma?', ['Sim', 'Não'], horizontal=True)

with col2:
    st.subheader("Alimentação e Atividade Física")
    cons_alim_caloricos = st.radio(
        'Consumo frequente de alimentos calóricos (FAVC)?',
        ['Sim', 'Não'], horizontal=True
    )
    freq_cons_veg = st.slider('Frequência de consumo de vegetais (FCVC)', 1.0, 3.0, 2.0, step=1.0, help="1: Nunca, 2: Às vezes, 3: Sempre")
    num_refeicoes = st.slider('Número de refeições principais diárias', 1.0, 5.0, 3.0, step=1.0)
    cons_lanches = st.select_slider(
        'Consumo de lanches entre refeições (CAEC)',
        options=['Não', 'As_vezes', 'Frequentemente', 'Sempre'],
        value='As_vezes'
    )
    cons_agua = st.slider('Consumo diário de água (Litros)', 1.0, 4.0, 2.0, step=1.0)
    cons_alcool = st.select_slider(
        'Consumo de bebida alcoólica (CALC)',
        options=['Não', 'As_vezes', 'Frequentemente', 'Sempre'],
        value='Não'
    )

st.subheader("Rotina Diária")
monitor_calorias = st.radio('Faz monitoramento de calorias ingeridas?', ['Sim', 'Não'], horizontal=True)
freq_ativ_fisica = st.slider('Frequência de atividade física semanal (FAF)', 0.0, 7.0, 2.0, step=1.0, help="Dias por semana")
tempo_telas = st.slider('Tempo diário em dispositivos eletrônicos (TUE)', 0.0, 10.0, 2.0, step=0.5, help="Horas por dia")
meio_transporte = st.selectbox(
    'Meio de transporte habitual (MTRANS)',
    # CORRIGIDO: Valores exatamente como no dicionário
    ['Transporte_publico', 'Carro', 'A_pe', 'Motocicleta', 'Bicicleta']
)

# --- BOTÃO DE PREVISÃO E LÓGICA ---
if st.button('**Calcular Nível de Obesidade**', use_container_width=True, type="primary"):
    if not MODEL_LOADED:
        st.error("O modelo não está carregado. Não é possível fazer a previsão.")
    else:
        # CORRIGIDO: As colunas que são float no seu CSV precisam ser float aqui também.
        dados_usuario = pd.DataFrame({
            'Sexo_biologico': [sexo],
            'Idade': [idade],
            'Altura': [altura],
            'Peso': [peso],
            'Historico_familiar_excesso_peso': [hist_familiar],
            'Consumo_frequente_alimentos_caloricos': [cons_alim_caloricos],
            'Frequencia_consumo_vegetais': [float(freq_cons_veg)],
            'Numero_refeicoes_principais': [float(num_refeicoes)],
            'Consumo_lanches_entre_refeicoes': [cons_lanches],
            'Habito_fumar': [fumo],
            'Consumo_diario_agua': [float(cons_agua)],
            'Monitoramento_ingestao_calorica': [monitor_calorias],
            'Frequencia_atividade_fisica_semanal': [float(freq_ativ_fisica)],
            'Tempo_diario_dispositivos_eletronicos': [float(tempo_telas)],
            'Consumo_bebida_alcoolica': [cons_alcool],
            'Meio_transporte_habitual': [meio_transporte]
        })
        
        st.subheader("Debug: Dados enviados para o modelo")
        st.dataframe(dados_usuario)
        st.write(dados_usuario.dtypes.astype(str))

        try:
            predicao_codificada = pipeline.predict(dados_usuario)
            resultado_legivel = target_encoder.inverse_transform(predicao_codificada)
            
            st.success(f'### O nível de obesidade previsto é: **{resultado_legivel[0]}**')

            imc_calculado = peso / (altura ** 2)
            st.info(f"O IMC calculado para os dados inseridos é: **{imc_calculado:.2f}**")

        except Exception as e:
            st.error(f"Ocorreu um erro durante a predição: {e}")
            st.error("Verifique a tabela de 'Debug' acima e compare com os dados de treino. Há alguma inconsistência nos nomes das categorias?")
