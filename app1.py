# ====================================================================
# SMARTBUILDER IA - ELLENCO ENGENHARIA
# Sistema Integrado de Inteligência Técnica e Operacional
# ====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Algoritmos e Métricas de Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix

from openai import OpenAI

# ====================================================================
# CONFIGURAÇÃO DA PÁGINA
# ====================================================================
st.set_page_config(page_title="Ellenco - SmartBuilder IA", page_icon="🏗️", layout="wide")

# ====================================================================
# CACHE E MODELOS (BOAS PRÁTICAS DE PERFORMANCE)
# ====================================================================
@st.cache_data
def carregar_dados_regressao():
    try:
        return pd.read_csv('concreto_resistencia.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def carregar_dados_classificacao():
    try:
        return pd.read_csv('inspecao_seguranca.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def carregar_dados_cluster():
    try:
        return pd.read_csv('gestao_canteiros.csv')
    except FileNotFoundError:
        return None

# ====================================================================
# MENU LATERAL
# ====================================================================
st.sidebar.image("ellenco.png", use_container_width=True)
st.sidebar.title("🏗️ SmartBuilder IA")
st.sidebar.caption("Plataforma de Engenharia Orientada a Dados")

menu = st.sidebar.radio(
    "Módulos Operacionais:",
    [
        "🏠 Início", 
        "📈 Dosagem & Resistência (Concreto)", 
        "🚦 Patologia & Risco Estrutural", 
        "📊 Clusterização & Performance de Obras", 
        "💬 Assistente Técnico de Manutenção"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("Compatível com diretrizes ABNT NBR 12655 / NBR 6118.")

# ====================================================================
# MÓDULO 0: INÍCIO
# ====================================================================
if menu == "🏠 Início":
    st.title("Sistema de Gestão Preditiva e Manutenção - Ellenco")
    st.markdown("""
    Este ambiente integra **Ciência de Dados** e normas técnicas de **Engenharia Civil**:
    
    * **Previsão de $f_{ck}$:** Modelagem não-linear baseada na relação $a/c$ (Lei de Abrams) e tempo de cura.
    * **Inspeção Estrutural:** Avaliação de risco baseada em limites normativos de abertura de fissura e perda de seção de aço.
    * **Auditoria de Canteiros:** Clusterização multivariável normalizada para comparação justa entre frentes de obra.
    * **Manutenção Especializada:** Chatbot assistente para resolução rápida de códigos de falha de equipamentos pesados.
    """)

# ====================================================================
# MÓDULO 1: REGRESSÃO (CONCRETO / LEI DE ABRAMS)
# ====================================================================
elif menu == "📈 Dosagem & Resistência (Concreto)":
    st.title("📈 Previsão Tecnológica da Resistência do Concreto ($f_c$)")
    st.caption("Modelagem não-linear com Random Forest considerando Fator Água/Cimento ($a/c$)")
    
    df_reg = carregar_dados_regressao()
    
    if df_reg is None:
        st.error("⚠️ Arquivo 'concreto_resistencia.csv' não encontrado.")
    else:
        aba_dados, aba_simulador, aba_metricas = st.tabs(["📂 Dados de Ensaio", "⚙️ Simulador de Traço", "📊 Validação do Modelo"])
        
        # Engenharia de Features (Física dos Materiais)
        df_reg['Fator_AC'] = df_reg['Agua_L'] / df_reg['Cimento_kg']
        df_reg['Log_Cura'] = np.log1p(df_reg['Tempo_Cura_dias'])
        
        features = ['Cimento_kg', 'Agua_L', 'Aditivo_ml', 'Tempo_Cura_dias', 'Fator_AC', 'Log_Cura']
        X = df_reg[features]
        y = df_reg['Resistencia_MPa']
        
        # Divisão Treino e Teste para evitar overfitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        modelo_concreto = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        modelo_concreto.fit(X_train, y_train)
        
        with aba_dados:
            st.markdown("### Resultados de Ruptura de Corpos de Prova")
            st.dataframe(df_reg, use_container_width=True)
            
        with aba_simulador:
            col1, col2 = st.columns(2)
            with col1:
                cimento = st.slider("Consumo de Cimento (kg/m³)", 200, 500, 320, step=10)
                agua = st.slider("Água de Amassamento (Litros)", 140, 240, 175, step=5)
                fator_ac_calc = agua / cimento
                st.info(f"🧪 **Fator Água/Cimento ($a/c$) Calculado:** `{fator_ac_calc:.2f}`")
                
            with col2:
                aditivo = st.slider("Aditivo Plastificante (ml/kg de cimento)", 0, 50, 15)
                idade_cura = st.selectbox("Idade de Controle / Cura (dias)", [3, 7, 14, 28, 56, 90], index=3)
                
            if st.button("Calcular Resistência Estimada", type="primary"):
                novo_ponto = pd.DataFrame({
                    'Cimento_kg': [cimento],
                    'Agua_L': [agua],
                    'Aditivo_ml': [aditivo],
                    'Tempo_Cura_dias': [idade_cura],
                    'Fator_AC': [fator_ac_calc],
                    'Log_Cura': [np.log1p(idade_cura)]
                })
                previsao = modelo_concreto.predict(novo_ponto)[0]
                
                col_res1, col_res2 = st.columns(2)
                col_res1.success(f"### Resistência Prevista: **{previsao:.2f} MPa**")
                if idade_cura == 28:
                    col_res2.metric("Classe Estimada", f"C{int(np.floor(previsao))}")
                    
        with aba_metricas:
            y_pred_test = modelo_concreto.predict(X_test)
            r2 = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Aderência Real (R² em Teste)", f"{r2 * 100:.1f}%")
            col_m2.metric("Margem Média de Erro (MAE)", f"± {mae:.2f} MPa")
            
            fig = px.scatter(
                x=y_test, y=y_pred_test,
                labels={'x': 'Resistência Real em Laboratório (MPa)', 'y': 'Previsão do Modelo (MPa)'},
                title="Curva de Aferição: Real vs. Previsto (Base de Teste Não Vista)",
                color_discrete_sequence=['#0055A5']
            )
            fig.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
            st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# MÓDULO 2: CLASSIFICAÇÃO (RISCO ESTRUTURAL / NBR 6118)
# ====================================================================
elif menu == "🚦 Patologia & Risco Estrutural":
    st.title("🚦 Diagnóstico Patológico e Avaliação de Risco Estrutural")
    
    df_class = carregar_dados_classificacao()
    if df_class is None:
        st.error("⚠️ Arquivo 'inspecao_seguranca.csv' não encontrado.")
    else:
        aba_dados, aba_semaforo, aba_metricas = st.tabs(["📂 Histórico de Vistorias", "🚨 Simulador de Inspeção", "📊 Matriz de Decisão"])
        
        X_c = df_class[['Fissura_mm', 'Corrosao_mm', 'Idade_Estrutura_anos']]
        y_c = df_class['Status_Risco']
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.3, random_state=42, stratify=y_c)
        
        # Árvore com profundidade controlada para evitar memorização indevida
        modelo_patologia = DecisionTreeClassifier(max_depth=4, min_samples_split=4, random_state=42)
        modelo_patologia.fit(X_train_c, y_train_c)
        
        with aba_dados:
            df_view = df_class.copy()
            labels_risco = {0: "🟢 Baixo / Seguro", 1: "🟡 Moderado / Monitorar", 2: "🔴 Crítico / Interdição"}
            df_view['Diagnóstico'] = df_view['Status_Risco'].map(labels_risco)
            st.dataframe(df_view, use_container_width=True)
            
        with aba_semaforo:
            st.markdown("#### Parâmetros Coletados em Campo:")
            c1, c2, c3 = st.columns(3)
            with c1:
                fissura = st.number_input("Abertura Máxima de Fissura (mm)", min_value=0.0, max_value=10.0, value=0.3, step=0.05, help="NBR 6118 estabelece limite de 0.2 a 0.4mm")
            with c2:
                corrosao = st.number_input("Perda Estimada de Seção / Armadura (mm)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
            with c3:
                idade = st.number_input("Idade da Estrutura (anos)", min_value=0, max_value=120, value=10, step=1)
                
            if st.button("Classificar Risco Estrutural", type="primary"):
                amostra = pd.DataFrame({'Fissura_mm': [fissura], 'Corrosao_mm': [corrosao], 'Idade_Estrutura_anos': [idade]})
                risco_pred = modelo_patologia.predict(amostra)[0]
                
                if risco_pred == 0:
                    st.success("🟢 **ESTRUTURA CONFORME / MONITORAMENTO DE ROTINA**\nParâmetros dentro dos limites normativos.")
                elif risco_pred == 1:
                    st.warning("🟡 **ALERTA: PATOLOGIA EM EVOLUÇÃO**\nProgramar ensaio não-destrutivo (esclerometria/ultrassom) e estanqueidade.")
                else:
                    st.error("🔴 **RISCO CRÍTICO ESTRUTURAL: AÇÃO IMEDIATA**\nProvidenciar escoramento/interdição e projeto de reforço estrutural.")
                    
        with aba_metricas:
            y_pred_c = modelo_patologia.predict(X_test_c)
            acc = accuracy_score(y_test_c, y_pred_c)
            st.metric("Acurácia Real em Teste", f"{acc * 100:.1f}%")
            
            matriz = confusion_matrix(y_test_c, y_pred_c)
            nomes_cat = ["Seguro", "Monitorar", "Interditar"]
            fig_mat = px.imshow(
                matriz,
                x=nomes_cat, y=nomes_cat,
                labels=dict(x="Classificação da IA", y="Realidade de Campo", color="Contagem"),
                text_auto=True, color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_mat, use_container_width=True)

# ====================================================================
# MÓDULO 3: CLUSTERIZAÇÃO (CANTEIROS PADRONIZADOS COM STANDARDSCALER)
# ====================================================================
elif menu == "📊 Clusterização & Performance de Obras":
    st.title("📊 Agrupamento e Perfil de Canteiros de Obras")
    st.caption("K-Means com normalização estatística Z-Score (`StandardScaler`)")
    
    df_clust = carregar_dados_cluster()
    if df_clust is None:
        st.error("⚠️ Arquivo 'gestao_canteiros.csv' não encontrado.")
    else:
        df_clust['ID_Obra'] = ['Obra ' + str(i+1) for i in range(len(df_clust))]
        variaveis_cluster = ['Consumo_Energia_kWh', 'Desperdicio_Material_%', 'Horas_Atraso']
        
        # 1. Normalização Fundamental para não distorcer o K-Means
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clust[variaveis_cluster])
        
        # 2. Clusterização
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        df_clust['Cluster_ID'] = kmeans.fit_predict(X_scaled)
        
        mapa_perfis = {
            0: "🟢 Canteiro Eficiente / Alto Padrão",
            1: "🟡 Desvio Operacional (Atenção)",
            2: "🔴 Canteiro Crítico (Alto Desperdício/Atraso)"
        }
        df_clust['Perfil_Operacional'] = df_clust['Cluster_ID'].map(mapa_perfis)
        
        aba_dados_c, aba_mapa = st.tabs(["📂 Tabela de Indicadores", "🗺️ Mapa Multidimensional"])
        
        with aba_dados_c:
            st.dataframe(df_clust[['ID_Obra', 'Consumo_Energia_kWh', 'Desperdicio_Material_%', 'Horas_Atraso', 'Perfil_Operacional']], use_container_width=True)
            
        with aba_mapa:
            fig_cl = px.scatter_3d(
                df_clust,
                x='Consumo_Energia_kWh',
                y='Desperdicio_Material_%',
                z='Horas_Atraso',
                color='Perfil_Operacional',
                hover_name='ID_Obra',
                title="Distribuição 3D dos Canteiros (Energia x Perda x Atraso)",
                color_discrete_map={
                    "🟢 Canteiro Eficiente / Alto Padrão": "#2ca02c",
                    "🟡 Desvio Operacional (Atenção)": "#ff7f0e",
                    "🔴 Canteiro Crítico (Alto Desperdício/Atraso)": "#d62728"
                }
            )
            st.plotly_chart(fig_cl, use_container_width=True)

# ====================================================================
# MÓDULO 4: ASSISTENTE TÉCNICO (RAG / MANUTENÇÃO)
# ====================================================================
elif menu == "💬 Assistente Técnico de Manutenção":
    st.title("💬 Assistente Especialista em Manutenção de Frotas")
    st.caption("Suporte técnico imediato para operadores e mecânicos de campo")
    
    with st.expander("⚙️ Credenciais de Integração Azure OpenAI"):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("API Key", type="password")
        with col2:
            endpoint = st.text_input("Endpoint", value="https://marcelomaffeis-05082026-resource.services.ai.azure.com/openai/v1")
        modelo_azure = st.text_input("Deployment Name", value="gpt-4.1-mini")
        
    try:
        with open('manual_falhas_maquinas.txt', 'r', encoding='utf-8') as f:
            base_manual = f.read()
    except FileNotFoundError:
        base_manual = "MANUAL PADRÃO: Procedimentos de segurança operacional para escavadeiras, vibroacabadoras e usinas."

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = [
            {
                "role": "system",
                "content": f"""Você é o Engenheiro Especialista Chefe de Manutenção Mecânica da Ellenco.
                Responda com foco em segurança do trabalho, produtividade de campo e passos práticos.
                Utilize o manual abaixo como fonte prioritária de verdade:
                
                --- MANUAL ---
                {base_manual}
                """
            }
        ]

    for msg in st.session_state.mensagens:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    pergunta = st.chat_input("Ex: Rolo compactador com superaquecimento hidráulico, qual o procedimento?")
    if pergunta:
        if not api_key:
            st.warning("⚠️ Forneça a chave da API Azure nas configurações acima.")
        else:
            st.session_state.mensagens.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.markdown(pergunta)
                
            with st.chat_message("assistant"):
                with st.spinner("Consultando manual de frotas..."):
                    try:
                        cliente = OpenAI(base_url=endpoint, api_key=api_key)
                        resposta = cliente.chat.completions.create(
                            model=modelo_azure,
                            messages=st.session_state.mensagens,
                            temperature=0.2
                        )
                        conteudo = resposta.choices[0].message.content
                        st.markdown(conteudo)
                        st.session_state.mensagens.append({"role": "assistant", "content": conteudo})
                    except Exception as err:
                        st.error(f"Erro na conexão com Azure OpenAI: {err}")

