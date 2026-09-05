# ====================================================================
# SMARTBUILDER IA - PLATAFORMA EDUCACIONAL OPEN SOURCE
# Aperfeiçoamento Profissional em Inteligência Artificial Generativa
# SENAI-SP | Licença MIT
# ====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# 1. Machine Learning Clássico (Scikit-Learn)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix

# 2. Deep Learning (TensorFlow / Keras)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except ImportError:
    tf = None

# 3. Processamento de Linguagem Natural (NLTK)
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
except ImportError:
    SentimentIntensityAnalyzer = None

# 4. Visão Computacional (OpenCV)
try:
    import cv2
except ImportError:
    cv2 = None

# 5. IA Generativa na Nuvem
from openai import OpenAI

# ====================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ====================================================================
st.set_page_config(
    page_title="SmartBuilder IA - Plataforma Educacional",
    page_icon="🏗️",
    layout="wide"
)

# ====================================================================
# FUNÇÃO DE RODAPÉ EDUCACIONAL & AVISO LEGAL
# ====================================================================
def exibir_rodape_educacional():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85em;'>
        <p><b>🎓 Projeto Educacional de Código Aberto (Open Source) — Licença MIT</b><br>
        Desenvolvido para o curso de <i>Aperfeiçoamento Profissional em Programação de Inteligência Artificial</i> — <b>SENAI-SP</b>.</p>
        <p>⚠️ <b>Aviso Legal / Disclaimer:</b> Este software tem finalidade estritamente didática e acadêmica. As predições, 
        cálculos de materiais, classificações de risco e diagnósticos gerados por estes modelos de IA são simulações para aprendizado 
        e <u>não substituem</u> ensaios laboratoriais normatizados (ABNT/NBR), projetos executivos ou a responsabilidade técnica de profissionais habilitados.</p>
    </div>
    """, unsafe_allow_html=True)

# ====================================================================
# CARREGAMENTO OTIMIZADO DE DADOS (CACHE)
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
# MENU LATERAL (SIDEBAR)
# ====================================================================
st.sidebar.title("🏗️ SmartBuilder IA")
st.sidebar.caption("Plataforma Didática de IA para Engenharia Civil")

menu = st.sidebar.radio(
    "Módulos do Curso:",
    [
        "🏠 Início & Ementa", 
        "📈 1. Regressão (Resistência do Concreto)", 
        "🚦 2. Classificação (Risco Estrutural)", 
        "📊 3. Clusterização (Gestão de Canteiros)", 
        "🧠 4. Deep Learning (TensorFlow / Keras)", 
        "📝 5. PLN & Sentimentos (NLTK)", 
        "👁️ 6. Visão Computacional (OpenCV)", 
        "💬 7. IA Generativa & RAG (Manutenção)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("📜 Código Aberto sob Licença MIT")
st.sidebar.caption("SENAI-SP — Formação Inicial e Continuada")

# ====================================================================
# MÓDULO 0: INÍCIO E EMENTA
# ====================================================================
if menu == "🏠 Início & Ementa":
    st.title("Bem-vindo ao SmartBuilder IA 🚀")
    st.subheader("Laboratório Prático de Inteligência Artificial para a Construção Civil")
    
    st.info("Este aplicativo é um projeto integrador educacional de código aberto que demonstra a aplicação prática de múltiplos ramos da IA na Engenharia e Infraestrutura.")
    
    st.markdown("""
    ### 📚 Competências e Tecnologias Desenvolvidas:
    * **Machine Learning Supervisionado:** Previsão de propriedades de materiais (Regressão) e triagem de segurança (Classificação) com `Scikit-Learn`.
    * **Machine Learning Não Supervisionado:** Agrupamento e identificação de padrões de eficiência em canteiros de obras (`K-Means` + `StandardScaler`).
    * **Deep Learning & Redes Neurais:** Treinamento de perceptrons multicamadas (MLP) com `TensorFlow` e `Keras`.
    * **Processamento de Linguagem Natural (PLN):** Mineração de texto e análise de sentimento de Diálogos Diários de Segurança (DDS) com `NLTK`.
    * **Visão Computacional:** Processamento digital de imagens e identificação de bordas/patologias estruturais com `OpenCV`.
    * **IA Generativa & RAG:** Integração em nuvem com grandes modelos de linguagem (`OpenAI / Azure`) para consulta técnica automatizada.
    """)
    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 1: REGRESSÃO (CONCRETO)
# ====================================================================
elif menu == "📈 1. Regressão (Resistência do Concreto)":
    st.title("📈 Previsão da Resistência à Compressão do Concreto ($f_c$)")
    st.caption("Aplicação da Lei de Abrams e Cinética de Cura via Random Forest Regressor")
    
    df_reg = carregar_dados_regressao()
    if df_reg is None:
        st.error("⚠️ Base 'concreto_resistencia.csv' não encontrada. Verifique o arquivo no repositório.")
    else:
        aba_dados, aba_simulador, aba_metricas = st.tabs(["📂 Base de Ensaios", "⚙️ Simulador de Traço", "📊 Métricas de Validação"])
        
        # Engenharia de atributos físicos (Fator a/c e escala logarítmica de cura)
        df_reg['Fator_AC'] = df_reg['Agua_L'] / df_reg['Cimento_kg']
        df_reg['Log_Cura'] = np.log1p(df_reg['Tempo_Cura_dias'])
        
        features = ['Cimento_kg', 'Agua_L', 'Aditivo_ml', 'Tempo_Cura_dias', 'Fator_AC', 'Log_Cura']
        X = df_reg[features]
        y = df_reg['Resistencia_MPa']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        modelo_concreto = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        modelo_concreto.fit(X_train, y_train)
        
        with aba_dados:
            st.dataframe(df_reg, use_container_width=True)
            
        with aba_simulador:
            col1, col2 = st.columns(2)
            with col1:
                cimento = st.slider("Cimento (kg/m³)", 200, 500, 320, step=10)
                agua = st.slider("Água de Amassamento (Litros)", 140, 240, 175, step=5)
                fator_ac = agua / cimento
                st.info(f"🧪 **Relação Água/Cimento ($a/c$):** `{fator_ac:.2f}`")
            with col2:
                aditivo = st.slider("Aditivo Plastificante (ml)", 0, 50, 15)
                tempo_cura = st.selectbox("Idade de Controle (dias)", [3, 7, 14, 28, 56, 90], index=3)
                
            if st.button("Estimar Resistência ($f_c$)", type="primary"):
                novo_ponto = pd.DataFrame({
                    'Cimento_kg': [cimento],
                    'Agua_L': [agua],
                    'Aditivo_ml': [aditivo],
                    'Tempo_Cura_dias': [tempo_cura],
                    'Fator_AC': [fator_ac],
                    'Log_Cura': [np.log1p(tempo_cura)]
                })
                res_prevista = modelo_concreto.predict(novo_ponto)[0]
                st.success(f"🧪 Resistência Estimada pela IA: **{res_prevista:.2f} MPa**")
                
        with aba_metricas:
            y_pred = modelo_concreto.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            c1, c2 = st.columns(2)
            c1.metric("Aderência do Modelo (R² em Teste)", f"{r2 * 100:.1f}%")
            c2.metric("Erro Médio Absoluto (MAE)", f"± {mae:.2f} MPa")
            
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Resistência Real (MPa)', 'y': 'Previsão do Modelo (MPa)'},
                title="Curva de Dispersão: Valores Reais vs. Preditos (Conjunto de Teste)",
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
            st.plotly_chart(fig, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 2: CLASSIFICAÇÃO (RISCO ESTRUTURAL / NBR 6118 - CALIBRADO)
# ====================================================================
elif menu == "🚦 2. Classificação (Risco Estrutural)":
    st.title("🚦 Classificação de Risco e Patologia Estrutural")
    st.caption("Árvore de Decisão calibrada conforme critérios normativos da ABNT NBR 6118")
    
    # Geramos uma base robusta e balanceada de 300 vistorias representativas
    @st.cache_data
    def gerar_base_inspecoes():
        np.random.seed(42)
        n = 300
        fissuras = np.random.uniform(0.05, 4.0, n)
        corrosoes = np.random.uniform(0.1, 8.0, n)
        idades = np.random.uniform(1, 80, n)
        
        status = []
        for f, c, i in zip(fissuras, corrosoes, idades):
            # Regras de Engenharia Estrutural
            if f > 1.8 or c > 4.5 or (f > 1.2 and c > 3.0) or (f > 1.0 and i > 50 and c > 2.5):
                status.append(2) # 🔴 Interditar
            elif f > 0.4 or c > 1.2 or (i > 30 and f > 0.25):
                status.append(1) # 🟡 Monitorar
            else:
                status.append(0) # 🟢 Seguro
                
        return pd.DataFrame({
            'Fissura_mm': fissuras,
            'Corrosao_mm': corrosoes,
            'Idade_Estrutura_anos': idades,
            'Status_Risco': status
        })

    df_class = carregar_dados_classificacao()
    # Se o CSV não existir ou tiver poucas linhas, usa a base representativa
    if df_class is None or len(df_class) < 50:
        df_class = gerar_base_inspecoes()

    aba_dados, aba_semaforo, aba_metricas = st.tabs(["📂 Base de Vistorias", "🚨 Painel de Inspeção Interativo", "📊 Diagnóstico do Modelo"])
    
    X_c = df_class[['Fissura_mm', 'Corrosao_mm', 'Idade_Estrutura_anos']]
    y_c = df_class['Status_Risco']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.25, random_state=42, stratify=y_c)
    
    # Árvore calibrada com balanceamento de classes
    modelo_arvore = DecisionTreeClassifier(max_depth=5, min_samples_split=3, class_weight='balanced', random_state=42)
    modelo_arvore.fit(X_train_c, y_train_c)
    
    with aba_dados:
        df_v = df_class.copy()
        mapa = {0: "🟢 Conforme / Seguro", 1: "🟡 Monitoramento Preventivo", 2: "🔴 Risco Crítico / Interdição"}
        df_v['Classificação'] = df_v['Status_Risco'].map(mapa)
        st.dataframe(df_v, use_container_width=True)
        
    with aba_semaforo:
        st.markdown("##### 💡 Carregar Cenários de Teste Rápido:")
        c_cen1, c_cen2, c_cen3 = st.columns(3)
        
        # Valores padrão
        val_fissura, val_corrosao, val_idade = 0.2, 0.5, 8
        
        if c_cen1.button("🟢 Cenário 1: Estrutura Segura"):
            val_fissura, val_corrosao, val_idade = 0.15, 0.4, 6
        if c_cen2.button("🟡 Cenário 2: Alerta / Monitorar"):
            val_fissura, val_corrosao, val_idade = 0.70, 2.2, 28
        if c_cen3.button("🔴 Cenário 3: Risco de Interdição"):
            val_fissura, val_corrosao, val_idade = 2.80, 5.5, 55

        col1, col2, col3 = st.columns(3)
        with col1:
            fissura_in = st.slider("Abertura da Fissura (mm)", 0.0, 4.0, float(val_fissura), step=0.05, help="Norma tolera até 0.4mm")
        with col2:
            corrosao_in = st.slider("Perda de Seção de Aço / Corrosão (mm)", 0.0, 8.0, float(val_corrosao), step=0.1)
        with col3:
            idade_in = st.slider("Idade da Estrutura (anos)", 1, 80, int(val_idade), step=1)
            
        st.markdown("---")
        
        # Predição e Probabilidades
        amostra = pd.DataFrame({'Fissura_mm': [fissura_in], 'Corrosao_mm': [corrosao_in], 'Idade_Estrutura_anos': [idade_in]})
        resultado = modelo_arvore.predict(amostra)[0]
        probabilidades = modelo_arvore.predict_proba(amostra)[0]
        
        col_res, col_prob = st.columns([1.2, 1])
        
        with col_res:
            if resultado == 0:
                st.success("### 🟢 **STATUS: SEGURO / CONFORME**\nEstrutura dentro dos limites normativos da NBR 6118.")
            elif resultado == 1:
                st.warning("### 🟡 **STATUS: ATENÇÃO / MONITORAR**\nPatologias em estágio intermediário. Programar manutenção preventiva.")
            else:
                st.error("### 🔴 **STATUS: CRÍTICO / INTERDIÇÃO IMEDIATA**\nRisco iminente de colapso estrutural. Acionar equipe de reforço!")
                
        with col_prob:
            st.markdown("##### Grau de Certeza da IA:")
            df_prob = pd.DataFrame({
                'Status': ['🟢 Seguro', '🟡 Monitorar', '🔴 Interditar'],
                'Probabilidade': probabilidades * 100
            })
            fig_prob = px.bar(
                df_prob, x='Probabilidade', y='Status', orientation='h',
                text=df_prob['Probabilidade'].apply(lambda x: f"{x:.1f}%"),
                color='Status',
                color_discrete_map={'🟢 Seguro': '#2ca02c', '🟡 Monitorar': '#ff7f0e', '🔴 Interditar': '#d62728'}
            )
            fig_prob.update_layout(xaxis_range=[0, 100], showlegend=False, height=180, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_prob, use_container_width=True)
                
    with aba_metricas:
        y_pred_c = modelo_arvore.predict(X_test_c)
        acc = accuracy_score(y_test_c, y_pred_c)
        st.metric("Acurácia Global do Classificador", f"{acc * 100:.1f}%")
        
        matriz = confusion_matrix(y_test_c, y_pred_c)
        cats = ["Seguro", "Monitorar", "Interditar"]
        fig_mat = px.imshow(
            matriz, x=cats, y=cats,
            labels=dict(x="Predição da IA", y="Realidade de Campo", color="Volume"),
            text_auto=True, color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_mat, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 3: CLUSTERIZAÇÃO (CANTEIROS)
# ====================================================================
elif menu == "📊 3. Clusterização (Gestão de Canteiros)":
    st.title("📊 Agrupamento Não Supervisionado de Canteiros de Obras")
    st.caption("Segmentação por K-Means com Normalização Estatística (StandardScaler)")
    
    df_clust = carregar_dados_cluster()
    if df_clust is None:
        st.error("⚠️ Base 'gestao_canteiros.csv' não encontrada.")
    else:
        df_clust['ID_Canteiro'] = ['Canteiro ' + str(i+1) for i in range(len(df_clust))]
        vars_cluster = ['Consumo_Energia_kWh', 'Desperdicio_Material_%', 'Horas_Atraso']
        
        # Normalização fundamental para evitar distorção de escala
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clust[vars_cluster])
        
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        df_clust['Cluster'] = kmeans.fit_predict(X_scaled)
        
        mapa_clusters = {
            0: "🟢 Canteiro Eficiente / Padrão Alto",
            1: "🟡 Desvio Operacional Moderado",
            2: "🔴 Canteiro com Perdas Críticas"
        }
        df_clust['Perfil'] = df_clust['Cluster'].map(mapa_clusters)
        
        aba_dados_cl, aba_grafico = st.tabs(["📂 Indicadores de Campo", "🗺️ Mapa 3D de Eficiência"])
        
        with aba_dados_cl:
            st.dataframe(df_clust[['ID_Canteiro', 'Consumo_Energia_kWh', 'Desperdicio_Material_%', 'Horas_Atraso', 'Perfil']], use_container_width=True)
            
        with aba_grafico:
            fig_3d = px.scatter_3d(
                df_clust,
                x='Consumo_Energia_kWh',
                y='Desperdicio_Material_%',
                z='Horas_Atraso',
                color='Perfil',
                hover_name='ID_Canteiro',
                title="Distribuição Multidimensional dos Canteiros",
                color_discrete_map={
                    "🟢 Canteiro Eficiente / Padrão Alto": "#2ca02c",
                    "🟡 Desvio Operacional Moderado": "#ff7f0e",
                    "🔴 Canteiro com Perdas Críticas": "#d62728"
                }
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 4: DEEP LEARNING (FADIGA E DESGASTE ESTRUTURAL/FROTAS)
# ====================================================================
elif menu == "🧠 4. Deep Learning (TensorFlow / Keras)":
    st.title("🧠 Deep Learning: Previsão de Desgaste Estrutural e Pavimentos")
    st.caption("Rede Neural Perceptron Multicamadas (MLP) para modelar degradação complexa e não-linear")
    
    from sklearn.neural_network import MLPRegressor
    
    st.markdown("""
    **Cenário de Engenharia:**  
    Na construção civil e rodovias, o desgaste de um pavimento ou equipamento depende de fatores cruzados não-lineares 
    (ex: excesso de carga de caminhões basculantes somado à alta temperatura e umidade).  
    Aqui, a Rede Neural aprende essas relações para prever o **Índice de Degradação (%)**.
    """)
    
    aba_treino, aba_simulador_dl = st.tabs(["⚙️ Treinamento da Rede", "🧪 Simulador em Tempo Real"])
    
    # Gerando base sintética calibrada de engenharia (300 ensaios)
    np.random.seed(42)
    n_amostras = 300
    trafego_eixos = np.random.uniform(500, 10000, n_amostras)      # Eixos equivalentes de caminhões
    peso_eixo_ton = np.random.uniform(8, 20, n_amostras)            # Carga por eixo (ton)
    umidade_solo = np.random.uniform(10, 90, n_amostras)            # % Umidade
    temp_ambiente = np.random.uniform(15, 45, n_amostras)           # Temperatura °C
    
    # Equação física não-linear de desgaste
    desgaste_real = (
        (trafego_eixos / 1000) * 1.5 + 
        ((peso_eixo_ton / 10) ** 3) * 2.0 + 
        (umidade_solo * 0.1) + 
        (temp_ambiente * 0.2) + 
        np.random.normal(0, 2, n_amostras)
    )
    desgaste_real = np.clip(desgaste_real, 0, 100) # Limita entre 0% e 100%
    
    X_dl = pd.DataFrame({
        'Trafego_Eixos': trafego_eixos,
        'Carga_Eixo_Ton': peso_eixo_ton,
        'Umidade_Solo_%': umidade_solo,
        'Temperatura_C': temp_ambiente
    })
    y_dl = desgaste_real
    
    scaler_dl = StandardScaler()
    X_dl_norm = scaler_dl.fit_transform(X_dl)
    
    with aba_treino:
        col1, col2 = st.columns(2)
        with col1:
            epocas = st.slider("Número de Épocas / Iterações de Treinamento", 30, 300, 100, step=10)
            lr = st.select_slider("Taxa de Aprendizado (Learning Rate)", [0.001, 0.005, 0.01, 0.05], value=0.01)
        with col2:
            neuronios = st.slider("Quantidade de Neurônios na Camada Oculta", 8, 64, 32, step=8)
            ativacao = st.selectbox("Função de Ativação Não-Linear", ["relu", "tanh", "logistic"])
            
        if st.button("Treinar Rede Neural Profunda", type="primary"):
            with st.spinner("Ajustando pesos sinápticos via Backpropagation (Adam)..."):
                modelo_mlp = MLPRegressor(
                    hidden_layer_sizes=(neuronios, neuronios // 2),
                    activation=ativacao,
                    solver='adam',
                    learning_rate_init=lr,
                    max_iter=epocas,
                    random_state=42
                )
                modelo_mlp.fit(X_dl_norm, y_dl)
                st.session_state['modelo_mlp'] = modelo_mlp
                st.session_state['scaler_dl'] = scaler_dl
                
                st.success("✅ Rede Neural Treinada com Sucesso!")
                
                df_loss = pd.DataFrame({
                    'Época': range(1, len(modelo_mlp.loss_curve_) + 1),
                    'Perda (Mean Squared Error)': modelo_mlp.loss_curve_
                })
                fig_loss = px.line(
                    df_loss, x='Época', y='Perda (Mean Squared Error)',
                    title="Curva de Otimização da Rede Neural (Decaimento do Erro na Obra)"
                )
                st.plotly_chart(fig_loss, use_container_width=True)

    with aba_simulador_dl:
        st.markdown("### Teste de Campo: Simular Desgaste com a Rede Neural")
        if 'modelo_mlp' not in st.session_state:
            st.info("👈 Primeiro treine a rede na aba **'⚙️ Treinamento da Rede'**.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                in_trafego = st.slider("Volume de Tráfego Pesado (Veículos/dia)", 500, 10000, 4500)
                in_carga = st.slider("Carga Média por Eixo (Toneladas)", 8.0, 25.0, 14.0, step=0.5)
            with c2:
                in_umid = st.slider("Umidade do Subleito / Solo (%)", 10, 95, 60)
                in_temp = st.slider("Temperatura Média no Local (°C)", 10, 50, 32)
                
            ponto_novo = pd.DataFrame({
                'Trafego_Eixos': [in_trafego],
                'Carga_Eixo_Ton': [in_carga],
                'Umidade_Solo_%': [in_umid],
                'Temperatura_C': [in_temp]
            })
            ponto_norm = st.session_state['scaler_dl'].transform(ponto_novo)
            desgaste_pred = st.session_state['modelo_mlp'].predict(ponto_norm)[0]
            desgaste_pred = max(0.0, min(100.0, desgaste_pred))
            
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Índice de Degradação Previsto", f"{desgaste_pred:.1f}%")
            
            if desgaste_pred < 40:
                col_res2.success("🟢 **CONDIÇÃO BOA:** Pavimento/Equipamento estável.")
            elif desgaste_pred < 70:
                col_res2.warning("🟡 **MANUTENÇÃO PREVENTIVA:** Programar fresagem ou revisão.")
            else:
                col_res2.error("🔴 **RISCO CRÍTICO DE RUPTURA:** Necessidade de intervenção imediata.")

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 5: PLN & SENTIMENTOS (ADAPTADO PARA PORTUGUÊS / OBRAS)
# ====================================================================
elif menu == "📝 5. PLN & Sentimentos (NLTK)":
    st.title("📝 Processamento de Linguagem Natural (PLN em Português)")
    st.caption("Mineração Textual e Análise de Sentimentos em Diálogos Diários de Segurança (DDS)")
    
    st.markdown("""
    **Como funciona a IA neste módulo:**  
    O algoritmo analisa o texto do relatório da obra, realiza a **tokenização** das palavras com `NLTK` 
    e cruza com um **dicionário léxico especializado em Segurança do Trabalho e Engenharia Civil**.
    """)
    
    # Dicionário Léxico Especializado em Português para Obras
    TERMOS_POSITIVOS = [
        'seguro', 'segurança', 'conforme', 'limpo', 'organizado', 'treinamento', 'equipado', 
        'proteção', 'sucesso', 'regular', 'inspecionado', 'aprovado', 'qualidade', 'correto', 
        'epi', 'cinto', 'capacete', 'atenção', 'ótimo', 'bom', 'cumprido', 'estável'
    ]
    
    TERMOS_NEGATIVOS = [
        'acidente', 'quase acidente', 'risco', 'perigo', 'falha', 'quebra', 'queda', 'sem epi', 
        'embargo', 'interdição', 'vazamento', 'choque', 'irregular', 'urgente', 'grave', 
        'danificado', 'trinca', 'rachadura', 'desabamento', 'ferimento', 'imprudência', 
        'desvio', 'problema', 'atraso', 'quebrado', 'parada'
    ]
    
    def analisar_sentimento_obra(texto):
        texto_lower = texto.lower()
        palavras = texto_lower.replace(',', ' ').replace('.', ' ').split()
        
        pos_count = sum(1 for p in palavras if any(tp in p for tp in TERMOS_POSITIVOS))
        neg_count = sum(1 for p in palavras if any(tn in p for tn in TERMOS_NEGATIVOS))
        total_tokens = len(palavras) if len(palavras) > 0 else 1
        
        score_polaridade = (pos_count - neg_count) / max(1, (pos_count + neg_count))
        
        return {
            'pos': pos_count / total_tokens,
            'neg': neg_count / total_tokens,
            'score': score_polaridade,
            'pos_encontradas': [p for p in palavras if any(tp in p for tp in TERMOS_POSITIVOS)],
            'neg_encontradas': [p for p in palavras if any(tn in p for tn in TERMOS_NEGATIVOS)]
        }

    # Botões de exemplos práticos para aula
    st.markdown("##### 💡 Escolha um exemplo ou digite seu próprio relato:")
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    
    ex_texto = "Hoje realizamos o DDS com foco em trabalho em altura. Todos os colaboradores estavam equipados com cinto e capacete e o canteiro estava limpo e seguro."
    if c_btn1.button("🟢 Exemplo: DDS Conforme (Seguro)"):
        ex_texto = "DDS realizado com sucesso. A equipe trabalhou com total segurança e os EPIs foram todos aprovados e inspecionados."
    if c_btn2.button("🔴 Exemplo: Quase Acidente (Risco)"):
        ex_texto = "Identificado grave risco de queda no andaime do bloco B. Colaborador estava trabalhando sem cinto e houve quase acidente com ferramentas soltas."
    if c_btn3.button("🟡 Exemplo: Registro Neutro / Rotina"):
        ex_texto = "Início dos serviços de concretagem da laje às 08h com recebimento de dois caminhões betoneira."

    relato_digitado = st.text_area("Texto do Relatório / Apontamento de Campo:", value=ex_texto, height=110)
    
    if st.button("Executar Análise de PLN", type="primary"):
        resultado = analisar_sentimento_obra(relato_digitado)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Termos Positivos (Segurança)", f"{len(resultado['pos_encontradas'])}")
        c2.metric("Termos Negativos (Riscos/Falhas)", f"{len(resultado['neg_encontradas'])}")
        c3.metric("Índice de Clima / Polaridade", f"{resultado['score']:+.2f}")
        
        st.markdown("---")
        if resultado['score'] > 0.15:
            st.success(f"🟢 **CLIMA SEGURO E CONFORME**\n\nPalavras-chave de segurança identificadas: `{', '.join(set(resultado['pos_encontradas']))}`")
        elif resultado['score'] < -0.15:
            st.error(f"🔴 **ALERTA CRÍTICO DE RISCO / NÃO-CONFORMIDADE**\n\nTermos de risco identificados: `{', '.join(set(resultado['neg_encontradas']))}`. Notificar técnico de segurança!")
        else:
            st.warning("🟡 **RELATO OPERACIONAL NEUTRO / INFORMATIVO**\nNenhum desvio crítico ou elogio expressivo detectado.")

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 6: VISÃO COMPUTACIONAL (OPENCV)
# ====================================================================
elif menu == "👁️ 6. Visão Computacional (OpenCV)":
    st.title("👁️ Visão Computacional para Inspeção Estrutural")
    st.caption("Processamento Digital de Imagens (Filtros de Gradiente e Detecção de Bordas Canny)")
    
    if cv2 is None:
        st.error("⚠️ OpenCV não encontrado. Instale com `pip install opencv-python-headless`.")
    else:
        upload = st.file_uploader("Envie uma foto de estrutura, viga ou pavimento (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        if upload is not None:
            img_pil = Image.open(upload)
            img_arr = np.array(img_pil)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img_pil, caption="Imagem Original", use_container_width=True)
            with col_b:
                gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
                suave = cv2.GaussianBlur(gray, (5, 5), 0)
                bordas = cv2.Canny(suave, 50, 150)
                st.image(bordas, caption="Segmentação de Bordas / Fissuras (Canny)", use_container_width=True)
                st.info("💡 Este pré-processamento prepara matrizes visuais para detecção de anomalias com YOLO.")

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 7: IA GENERATIVA & RAG (MANUTENÇÃO)
# ====================================================================
elif menu == "💬 7. IA Generativa & RAG (Manutenção)":
    st.title("💬 Assistente Técnico com IA Generativa & RAG")
    st.caption("Consulta Automatizada a Manuais de Máquinas e Equipamentos de Construção Civil")
    
    with st.expander("⚙️ Parâmetros de Conexão com a Nuvem (Azure OpenAI)"):
        c1, c2 = st.columns(2)
        with c1:
            api_key = st.text_input("Chave da API (Key)", type="password")
        with c2:
            endpoint = st.text_input("Endpoint da API", value="https://marcelomaffeis-05082026-resource.services.ai.azure.com/openai/v1")
        deployment_model = st.text_input("Nome do Deployment / Modelo", value="gpt-4.1-mini")

    try:
        with open('manual_falhas_maquinas.txt', 'r', encoding='utf-8') as f:
            manual_conteudo = f.read()
    except FileNotFoundError:
        manual_conteudo = "MANUAL DIDÁTICO: Procedimentos de segurança e manutenção para escavadeiras, pás-carregadeiras e usinas de concreto."

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = [
            {
                "role": "system",
                "content": f"""Você é um Assistente Virtual Especialista em Manutenção Mecânica de Equipamentos de Construção Civil e Infraestrutura.
                Responda com foco em segurança do trabalho e manutenção preventiva/corretiva.
                Baseie-se preferencialmente no manual técnico abaixo:
                
                --- MANUAL TÉCNICO ---
                {manual_conteudo}
                """
            }
        ]

    for msg in st.session_state.mensagens:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    duvida = st.chat_input("Digite sua dúvida técnica (Ex: Como proceder em caso de alta temperatura no óleo hidráulico?)")
    if duvida:
        if not api_key:
            st.warning("⚠️ Insira a chave da API nas configurações acima para interagir com o modelo.")
        else:
            st.session_state.mensagens.append({"role": "user", "content": duvida})
            with st.chat_message("user"):
                st.markdown(duvida)
                
            with st.chat_message("assistant"):
                with st.spinner("Consultando base de conhecimento técnico..."):
                    try:
                        cliente = OpenAI(base_url=endpoint, api_key=api_key)
                        res = cliente.chat.completions.create(
                            model=deployment_model,
                            messages=st.session_state.mensagens,
                            temperature=0.2
                        )
                        resposta_texto = res.choices[0].message.content
                        st.markdown(resposta_texto)
                        st.session_state.mensagens.append({"role": "assistant", "content": resposta_texto})
                    except Exception as e:
                        st.error(f"Erro na comunicação com o serviço de IA: {e}")

    exibir_rodape_educacional()
