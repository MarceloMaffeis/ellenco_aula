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
# MÓDULO 2: CLASSIFICAÇÃO (RISCO ESTRUTURAL)
# ====================================================================
elif menu == "🚦 2. Classificação (Risco Estrutural)":
    st.title("🚦 Classificação de Risco e Patologia Estrutural")
    st.caption("Tomada de Decisão baseada em Árvores de Decisão (Critérios NBR 6118)")
    
    df_class = carregar_dados_classificacao()
    if df_class is None:
        st.error("⚠️ Base 'inspecao_seguranca.csv' não encontrada.")
    else:
        aba_dados, aba_semaforo, aba_metricas = st.tabs(["📂 Vistorias Anteriores", "🚨 Painel de Inspeção", "📊 Diagnóstico Técnico"])
        
        X_c = df_class[['Fissura_mm', 'Corrosao_mm', 'Idade_Estrutura_anos']]
        y_c = df_class['Status_Risco']
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.3, random_state=42, stratify=y_c)
        
        modelo_arvore = DecisionTreeClassifier(max_depth=4, min_samples_split=4, random_state=42)
        modelo_arvore.fit(X_train_c, y_train_c)
        
        with aba_dados:
            df_v = df_class.copy()
            mapa = {0: "🟢 Conforme / Seguro", 1: "🟡 Monitoramento Preventivo", 2: "🔴 Risco Crítico / Interdição"}
            df_v['Classificação'] = df_v['Status_Risco'].map(mapa)
            st.dataframe(df_v, use_container_width=True)
            
        with aba_semaforo:
            col1, col2, col3 = st.columns(3)
            with col1:
                fissura_in = st.number_input("Abertura da Fissura (mm)", 0.0, 10.0, 0.3, step=0.05)
            with col2:
                corrosao_in = st.number_input("Perda de Seção / Corrosão (mm)", 0.0, 15.0, 1.0, step=0.1)
            with col3:
                idade_in = st.number_input("Idade da Estrutura (anos)", 1, 100, 15, step=1)
                
            if st.button("Avaliar Risco Estrutural", type="primary"):
                amostra = pd.DataFrame({'Fissura_mm': [fissura_in], 'Corrosao_mm': [corrosao_in], 'Idade_Estrutura_anos': [idade_in]})
                resultado = modelo_arvore.predict(amostra)[0]
                
                if resultado == 0:
                    st.success("🟢 **STATUS: SEGURO / CONFORME**\nParâmetros compatíveis com uso normal.")
                elif resultado == 1:
                    st.warning("🟡 **STATUS: ATENÇÃO / MONITORAR**\nRecomenda-se acompanhamento periódico de evolução.")
                else:
                    st.error("🔴 **STATUS: CRÍTICO / INTERDIÇÃO RECOMENDADA**\nNecessidade de intervenção emergencial e projeto de reforço.")
                    
        with aba_metricas:
            y_pred_c = modelo_arvore.predict(X_test_c)
            acc = accuracy_score(y_test_c, y_pred_c)
            st.metric("Acurácia do Modelo em Base de Teste", f"{acc * 100:.1f}%")
            
            matriz = confusion_matrix(y_test_c, y_pred_c)
            cats = ["Seguro", "Monitorar", "Interditar"]
            fig_mat = px.imshow(
                matriz, x=cats, y=cats,
                labels=dict(x="Predição da IA", y="Realidade", color="Volume"),
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
# MÓDULO 4: REDES NEURAIS ARTIFICIAIS (DEEP LEARNING / MLP)
# ====================================================================
elif menu == "🧠 4. Deep Learning (TensorFlow / Keras)":
    st.title("🧠 Redes Neurais Artificiais (Deep Learning)")
    st.caption("Treinamento de Redes Perceptron Multicamadas (MLP) com Função de Custo e Iterações")
    
    from sklearn.neural_network import MLPRegressor
    
    st.markdown("### Treinamento de Rede Neural para Previsão Operacional")
    
    col1, col2 = st.columns(2)
    with col1:
        epocas = st.slider("Número Máximo de Iterações / Épocas", 20, 200, 80, step=10)
        lr = st.select_slider("Taxa de Aprendizado (Learning Rate Inicial)", [0.001, 0.005, 0.01, 0.05], value=0.01)
    with col2:
        neuronios = st.slider("Neurônios na Camada Oculta", 8, 64, 16, step=8)
        ativacao = st.selectbox("Função de Ativação", ["relu", "tanh", "logistic"])

    if st.button("Treinar Rede Neural (MLP)", type="primary"):
        with st.spinner("Ajustando pesos sinápticos e treinando a rede neural..."):
            np.random.seed(42)
            X_s = np.random.rand(300, 3) * [1000, 100, 50]
            y_s = (0.0005 * X_s[:, 0] + 0.003 * X_s[:, 1] + 0.01 * X_s[:, 2]) * 10
            
            sc = StandardScaler()
            X_norm = sc.fit_transform(X_s)
            
            # Arquitetura da Rede Neural Perceptron Multicamadas
            rede_neural = MLPRegressor(
                hidden_layer_sizes=(neuronios, neuronios // 2),
                activation=ativacao,
                solver='adam',
                learning_rate_init=lr,
                max_iter=epocas,
                random_state=42
            )
            rede_neural.fit(X_norm, y_s)
            
            st.success("✅ Rede Neural Treinada com Sucesso!")
            
            # Gráfico de Perda (Loss Curve)
            df_loss = pd.DataFrame({
                'Iteração': range(1, len(rede_neural.loss_curve_) + 1),
                'Função de Perda (Loss)': rede_neural.loss_curve_
            })
            fig_loss = px.line(
                df_loss, x='Iteração', y='Função de Perda (Loss)',
                title="Curva de Convergência do Gradiente (Decaimento do Erro Quadrático)"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 5: PLN & SENTIMENTOS (NLTK)
# ====================================================================
elif menu == "📝 5. PLN & Sentimentos (NLTK)":
    st.title("📝 Processamento de Linguagem Natural (PLN)")
    st.caption("Mineração Textual e Análise de Sentimentos em Diálogos Diários de Segurança (DDS)")
    
    if SentimentIntensityAnalyzer is None:
        st.error("⚠️ NLTK não configurado. Execute `pip install nltk`.")
    else:
        sia = SentimentIntensityAnalyzer()
        st.markdown("### Auditoria Automatizada de Relatórios e Diálogos de Segurança")
        
        texto_padrao = "Hoje realizamos o DDS com foco em trabalho em altura. Todos os colaboradores estavam equipados com cinto tipo paraquedista e a área foi devidamente sinalizada."
        texto_usuario = st.text_area("Insira o relato do DDS ou anotação do canteiro:", value=texto_padrao, height=120)
        
        if st.button("Executar Análise de PLN", type="primary"):
            scores = sia.polarity_scores(texto_usuario)
            comp = scores['compound']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Índice Positivo", f"{scores['pos']*100:.1f}%")
            c2.metric("Índice Neutro", f"{scores['neu']*100:.1f}%")
            c3.metric("Índice Negativo / Alerta", f"{scores['neg']*100:.1f}%")
            
            if comp >= 0.05:
                st.success("🟢 **Relato em Conformidade:** Texto com indicativos claros de boas práticas e segurança.")
            elif comp <= -0.05:
                st.error("🔴 **Alerta de Inconformidade:** Termos associados a riscos, queixas, falhas ou perigos detectados.")
            else:
                st.warning("🟡 **Relato Informativo / Neutro.**")

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
