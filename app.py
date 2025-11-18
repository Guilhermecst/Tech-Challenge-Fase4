import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

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
# Mapeia os valores do modelo para valores legíveis (e vice-versa)
map_transporte = {'Transporte publico': 'Transporte_publico', 'Carro': 'Carro', 'A pe': 'A_pe', 'Motocicleta': 'Motocicleta', 'Bicicleta': 'Bicicleta'}
map_lanches_alcool = {'Não': 'Não', 'As vezes': 'As_vezes', 'Frequentemente': 'Frequentemente', 'Sempre': 'Sempre'}

# --- BARRA LATERAL ---
with st.sidebar:
    # ... (código da sidebar inalterado) ...
    st.title('Calculadora de Obesidade')
    st.markdown("Este aplicativo usa Machine Learning para prever o nível de obesidade.")
    st.markdown("---")
    st.subheader("Status do Modelo")

    MODEL_LOADED = False
    try:
        pipeline = joblib.load('modelo_svc.joblib')
        target_encoder = joblib.load('target_encoder_obesidade.joblib')
        st.success("Modelo carregado!")
        MODEL_LOADED = True
    except FileNotFoundError:
        st.error("Arquivos do modelo não encontrados.")
        st.warning("Execute o script de treinamento.")
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")

    st.markdown("---")
    st.info("Projeto de análise de dados em saúde.")


# --- INTERFACE PRINCIPAL ---
st.title('Preencha os campos para a previsão')

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
        freq_cons_veg = st.slider('Frequência de consumo de vegetais', 1, 3, 2, help="1: Nunca, 2: Às vezes, 3: Sempre")
        num_refeicoes = st.slider('Número de refeições principais diárias', 1, 5, 3)
        cons_lanches_display = st.select_slider(
            'Consumo de lanches entre refeições',
            options=list(map_lanches_alcool.keys()), # Usa as chaves do mapa (sem _)
            value='As vezes'
        )
        cons_agua = st.slider('Consumo diário de água (Litros)', 1, 4, 2)
        cons_alcool_display = st.select_slider(
            'Consumo de bebida alcoólica',
            options=list(map_lanches_alcool.keys()), # Usa as chaves do mapa (sem _)
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
            options=list(map_transporte.keys()) # Usa as chaves do mapa (sem _)
        )

# --- BOTÃO DE PREVISÃO E RESULTADO ---
if st.button('**✨ Calcular Nível de Obesidade**', use_container_width=True, type="primary"):
    if not MODEL_LOADED:
        st.error("O modelo não está carregado. Não é possível fazer a previsão.")
    else:
        # Converte os valores da interface de volta para o formato esperado pelo modelo
        cons_lanches_modelo = map_lanches_alcool[cons_lanches_display]
        cons_alcool_modelo = map_lanches_alcool[cons_alcool_display]
        meio_transporte_modelo = map_transporte[meio_transporte_display]

        dados_usuario = pd.DataFrame({
            'Sexo_biologico': [sexo], 'Idade': [idade], 'Altura': [altura], 'Peso': [peso],
            'Historico_familiar_excesso_peso': [hist_familiar],
            'Consumo_frequente_alimentos_caloricos': [cons_alim_caloricos],
            'Frequencia_consumo_vegetais': [float(freq_cons_veg)],
            'Numero_refeicoes_principais': [float(num_refeicoes)],
            'Consumo_lanches_entre_refeicoes': [cons_lanches_modelo], # Usa o valor convertido
            'Habito_fumar': [fumo],
            'Consumo_diario_agua': [float(cons_agua)],
            'Monitoramento_ingestao_calorica': [monitor_calorias],
            'Frequencia_atividade_fisica_semanal': [float(freq_ativ_fisica)],
            'Tempo_diario_dispositivos_eletronicos': [float(tempo_telas)],
            'Consumo_bebida_alcoolica': [cons_alcool_modelo], # Usa o valor convertido
            'Meio_transporte_habitual': [meio_transporte_modelo] # Usa o valor convertido
        })

        try:
            predicao_codificada = pipeline.predict(dados_usuario)
            resultado_com_underline = target_encoder.inverse_transform(predicao_codificada)
            
            # Remove o underline do resultado final
            resultado_final = resultado_com_underline[0].replace('_', ' ')
            
            imc_calculado = peso / (altura ** 2)

            st.markdown("---")
            st.header("Resultados da Análise")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="IMC Calculado", value=f"{imc_calculado:.2f}")
            with res_col2:
                # Exibe o resultado já formatado
                st.metric(label="Previsão do Nível de Obesidade", value=resultado_final)

        except Exception as e:
            st.error(f"Ocorreu um erro durante a predição: {e}")
            st.warning("Verifique se os dados inseridos são consistentes.")