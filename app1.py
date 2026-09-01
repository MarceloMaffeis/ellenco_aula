# ====================================================================
# IMPORTAÇÃO DE BIBLIOTECAS (O "Kit de Ferramentas" do Desenvolvedor)
# ====================================================================
import streamlit as st                  # Para criar a interface gráfica (Dashboards) na web
import pandas as pd                     # Para ler, manipular e analisar tabelas de dados (DataFrames)
import matplotlib.pyplot as plt         # Para criar gráficos básicos (não interativos)
import seaborn as sns                   # Para criar gráficos estatísticos mais bonitos (baseado no matplotlib)
import plotly.express as px             # Para criar gráficos avançados e interativos
import plotly.graph_objects as go       # Para adicionar camadas complexas aos gráficos do Plotly
from sklearn.linear_model import LinearRegression     # Algoritmo de IA para prever números (Regressão)
from sklearn.tree import DecisionTreeClassifier       # Algoritmo de IA para categorizar dados (Classificação)
from sklearn.cluster import KMeans                    # Algoritmo de IA para agrupar dados semelhantes (Clusterização)
from sklearn.metrics import r2_score, mean_absolute_error # Ferramentas para medir acertos em Regressão
from sklearn.metrics import accuracy_score, confusion_matrix # Ferramentas para medir acertos em Classificação
from openai import OpenAI               # Biblioteca para conectar com IAs Generativas (como o ChatGPT na Azure)

# ====================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ====================================================================
# Define o nome que aparece na aba do navegador, o ícone e o layout expandido para usar toda a tela
st.set_page_config(page_title="Sistema de Obras IA", page_icon="🏗️", layout="wide")

# ====================================================================
# MENU LATERAL (SIDEBAR) - NAVEGAÇÃO DO SISTEMA
# ====================================================================
st.sidebar.title("🏗️ SmartBuilder IA")
st.sidebar.markdown("Sistema Integrado de Gestão e Manutenção")

# st.sidebar.radio cria os botões para o usuário escolher a página
menu = st.sidebar.radio(
    "Selecione o Módulo:",
    [
        "🏠 Início", 
        "📈 Regressão (Concreto)", 
        "🚦 Classificação (Segurança)", 
        "📊 Clusterização (Canteiros)", 
        "💬 Assistente RAG (Manutenção)"
    ]
)

st.sidebar.markdown("---") # Cria uma linha divisória
st.sidebar.caption("SENAI - Projeto Final de Inteligência Artificial")

# ====================================================================
# MÓDULO 0: INÍCIO (TELA DE BOAS-VINDAS)
# ====================================================================
if menu == "🏠 Início":
    st.title("Bem-vindo ao SmartBuilder IA 🚀")
    st.info("Utilize o menu lateral para navegar entre os módulos de Inteligência Artificial.")
    st.markdown("""
    **O que este sistema faz?**
    * Previsão de resistência de materiais (Machine Learning Clássico)
    * Análise de risco estrutural (Classificação)
    * Descoberta de perfis de canteiros (Clusterização)
    * Chatbot especialista em manutenção (IA Generativa RAG)
    """)

# ====================================================================
# MÓDULO 1: REGRESSÃO (PREVENDO NÚMEROS CONTÍNUOS)
# ====================================================================
elif menu == "📈 Regressão (Concreto)":
    st.title("📈 Previsão de Resistência do Concreto")
    try:
        # Tenta carregar o banco de dados. Se o arquivo não existir, pula para o 'except'
        df_regressao = pd.read_csv('concreto_resistencia.csv')
        
        # st.tabs organiza o conteúdo em abas clicáveis para deixar a tela mais limpa
        aba_dados, aba_simulador, aba_metricas = st.tabs(["📂 Base de Dados", "⚙️ Simulador de IA", "📊 Confiabilidade da IA"])
        
        # --- PASSO 1: PREPARAÇÃO DA IA ---
        # Separamos os dados: 'X' são as "perguntas" (ingredientes) e 'y' é a "resposta" (resistência final)
        X = df_regressao[['Cimento_kg', 'Agua_L', 'Aditivo_ml', 'Tempo_Cura_dias']]
        y = df_regressao['Resistencia_MPa']
        
        # Inicializa o robô matemático e manda ele aprender (.fit) a relação entre ingredientes e resistência
        modelo_regressao = LinearRegression()
        modelo_regressao.fit(X, y)
        
        # --- ABA 1: MOSTRANDO A TABELA BRUTA ---
        with aba_dados:
            st.write("Visualização dos dados reais coletados no laboratório:")
            st.dataframe(df_regressao, use_container_width=True) # Exibe a tabela do Pandas no Streamlit
            
        # --- ABA 2: O SIMULADOR INTERATIVO ---
        with aba_simulador:
            st.write("Ajuste as proporções do traço para prever a resistência final do concreto.")
            
            # st.columns divide a tela em blocos lado a lado
            col1, col2 = st.columns(2)
            with col1:
                # Cria botões deslizantes para o usuário inserir os dados
                cimento_input = st.slider("Cimento (kg)", 200, 450, 300)
                agua_input = st.slider("Água (Litros)", 150, 220, 180)
            with col2:
                aditivo_input = st.slider("Aditivo (ml)", 0, 50, 20)
                # Cria um menu suspenso para escolher o tempo de cura
                cura_input = st.selectbox("Tempo de Cura (dias)", [7, 14, 28, 56], index=2)
                
            st.markdown("---")
            # Quando o botão for clicado, o bloco de código abaixo é executado
            if st.button("Calcular Resistência", type="primary"):
                # 1. Monta uma nova "tabelinha" apenas com os dados que o usuário digitou
                novo_traco = pd.DataFrame({
                    'Cimento_kg': [cimento_input], 
                    'Agua_L': [agua_input], 
                    'Aditivo_ml': [aditivo_input], 
                    'Tempo_Cura_dias': [cura_input]
                })
                # 2. Pede para a IA prever o resultado baseado no que ela aprendeu (.predict)
                nova_previsao = modelo_regressao.predict(novo_traco)
                # 3. Mostra o resultado na tela limitando a 2 casas decimais (.2f)
                st.success(f"🧪 A resistência estimada pela IA é de **{nova_previsao[0]:.2f} MPa**")
                
        # --- ABA 3: AUDITORIA E MÉTRICAS ---
        with aba_metricas:
            st.markdown("### Diagnóstico Técnico da Inteligência Artificial")
            st.write("Avaliação de desempenho comparando as previsões da IA com os resultados reais do laboratório.")
            
            # Para testar a IA, pedimos para ela prever tudo o que está na base de dados
            previsoes_totais = modelo_regressao.predict(X)
            
            # Calculamos a taxa de acerto global (R2) e a margem de erro média (MAE)
            taxa_acerto_r2 = r2_score(y, previsoes_totais)
            margem_erro_mae = mean_absolute_error(y, previsoes_totais)
            
            # Mostramos esses números em formato de 'cartões' em destaque (st.metric)
            col_met1, col_met2 = st.columns(2)
            col_met1.metric(label="Precisão do Modelo (R²)", value=f"{taxa_acerto_r2 * 100:.1f}%")
            col_met2.metric(label="Margem de Erro (MAE)", value=f"± {margem_erro_mae:.2f} MPa", delta_color="inverse")
            
            st.markdown("---")
            
            # Montamos uma tabela temporária para criar o gráfico comparativo
            df_metricas = pd.DataFrame({'Real': y, 'Previsto': previsoes_totais})
            
            # px.scatter cria o gráfico interativo de bolhas
            fig_reg = px.scatter(
                df_metricas, x='Real', y='Previsto', 
                title='Teste de Laboratório vs. Previsão da IA',
                labels={'Real': 'Resistência Real', 'Previsto': 'Resistência Prevista'},
                color_discrete_sequence=['#ef553b']
            )
            
            # Adicionamos uma linha guia. Se a bolha cai em cima da linha, a IA acertou 100%
            fig_reg.add_shape(type="line", line=dict(dash="dash", color="gray", width=2), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
            st.plotly_chart(fig_reg, use_container_width=True)

    except FileNotFoundError:
        st.error("⚠️ Erro: O arquivo 'concreto_resistencia.csv' não foi encontrado na mesma pasta do script.")

# ====================================================================
# MÓDULO 2: CLASSIFICAÇÃO (PREVENDO CATEGORIAS)
# ====================================================================
elif menu == "🚦 Classificação (Segurança)":
    st.title("🚦 Inspeção de Risco Estrutural")
    try:
        df_classificacao = pd.read_csv('inspecao_seguranca.csv')
        aba_dados, aba_semaforo, aba_metricas = st.tabs(["📂 Base de Dados", "🚨 Painel de Risco", "📊 Confiabilidade da IA"])
        
        # --- PASSO 1: PREPARAÇÃO DA IA ---
        # 'X' são as anomalias detectadas; 'y' é o nível de risco atribuído (0, 1 ou 2)
        X_class = df_classificacao[['Fissura_mm', 'Corrosao_mm', 'Idade_Estrutura_anos']]
        y_class = df_classificacao['Status_Risco']
        
        # Inicializa o modelo de Árvore de Decisão e realiza o treinamento (.fit)
        modelo_arvore = DecisionTreeClassifier(random_state=42) # random_state garante que o resultado será sempre o mesmo (reprodutível)
        modelo_arvore.fit(X_class, y_class)
        
        with aba_dados:
            st.write("Histórico de vistorias estruturais passadas:")
            df_visual = df_classificacao.copy()
            # Traduz os números (0, 1, 2) para palavras compreensíveis na tabela
            mapa_cores = {0: "🟢 Seguro", 1: "🟡 Monitorar", 2: "🔴 Interditar"}
            df_visual['Alerta'] = df_visual['Status_Risco'].map(mapa_cores)
            st.dataframe(df_visual, use_container_width=True)
            
        with aba_semaforo:
            st.write("Insira os dados da vistoria atual para avaliação de risco.")
            col1, col2, col3 = st.columns(3)
            with col1:
                # st.number_input permite digitar um número decimal com passos específicos (step)
                fissura_input = st.number_input("Fissura (mm)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
            with col2:
                corrosao_input = st.number_input("Corrosão (mm)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
            with col3:
                idade_input = st.number_input("Idade (anos)", min_value=1, max_value=100, value=15, step=1)
                
            st.markdown("---")
            if st.button("Avaliar Risco", type="primary"):
                # Cria a tabela com os dados digitados respeitando exatamente o nome das colunas do CSV
                nova_inspecao = pd.DataFrame({'Fissura_mm': [fissura_input], 'Corrosao_mm': [corrosao_input], 'Idade_Estrutura_anos': [idade_input]})
                
                # A IA analisa a tabela e devolve a previsão
                previsao_risco = modelo_arvore.predict(nova_inspecao)[0]
                
                # Regras de exibição condicional baseadas na resposta da IA
                if previsao_risco == 0:
                    st.success("🟢 **STATUS: SEGURO**")
                elif previsao_risco == 1:
                    st.warning("🟡 **STATUS: MONITORAR**")
                else:
                    st.error("🔴 **STATUS: INTERDITAR!**")
                    
        with aba_metricas:
            st.markdown("### Diagnóstico Técnico (Matriz de Confusão)")
            
            # Testa a IA gerando previsões para todos os dados conhecidos
            previsoes_class = modelo_arvore.predict(X_class)
            
            # Calcula quantos % a IA acertou globalmente
            acuracia = accuracy_score(y_class, previsoes_class)
            st.metric(label="Acurácia Geral", value=f"{acuracia * 100:.1f}%")
            
            # A matriz de confusão cruza o "Previsão" com o "Real" para vermos os Falsos Positivos/Negativos
            matriz = confusion_matrix(y_class, previsoes_class)
            categorias = ["Seguro", "Monitorar", "Interditar"]
            
            # px.imshow gera um gráfico de calor (Heatmap) interativo para a matriz
            fig_matriz = px.imshow(
                matriz,
                labels=dict(x="Previsão da IA", y="Realidade", color="Volume"),
                x=categorias, y=categorias, text_auto=True, color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_matriz, use_container_width=True)
            st.info("⚠️ Nota Técnica: A acurácia de 100% ocorre porque o modelo está sendo avaliado nos mesmos dados usados no treinamento (Base de Treino). Em produção, utilizaríamos a técnica 'train_test_split'.")
            
    except FileNotFoundError:
        st.error("⚠️ Arquivo 'inspecao_seguranca.csv' não encontrado.")

# ====================================================================
# MÓDULO 3: CLUSTERIZAÇÃO (AGRUPAMENTO POR COMPORTAMENTO)
# ====================================================================
elif menu == "📊 Clusterização (Canteiros)":
    st.title("📊 Análise e Perfil de Canteiros (Aprendizado Não Supervisionado)")
    try:
        df_cluster = pd.read_csv('gestao_canteiros.csv')
        
        # Criamos um nome fantasia (ID) para cada obra no gráfico
        df_cluster['ID_Obra'] = ['Obra ' + str(i+1) for i in range(len(df_cluster))]
        
        # --- PASSO 1: PREPARAÇÃO DA IA ---
        # Note que aqui não temos a variável 'y' (resposta). A IA vai descobrir os grupos sozinha.
        X_cluster = df_cluster[['Consumo_Energia_kWh', 'Desperdicio_Material_%', 'Horas_Atraso']]
        
        # Configuramos a IA (KMeans) para procurar exatamente 3 grupos de perfis semelhantes
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        
        # Treina e ao mesmo tempo já cria a nova coluna classificando cada obra (1, 2 ou 3)
        # .astype(str) transforma o número em texto para o Plotly separar bem as cores
        df_cluster['Grupo_IA'] = kmeans.fit_predict(X_cluster).astype(str) 
        
        # Extraímos os 'centroides' (os pontos médios matemáticos de cada grupo)
        centroides = kmeans.cluster_centers_
        
        aba_dados, aba_grafico = st.tabs(["📂 Base de Dados", "🗺️ Mapa de Clusters Interativo"])
        
        with aba_dados:
            st.dataframe(df_cluster, use_container_width=True)
            
        with aba_grafico:
            # Gráfico de dispersão interativo com Plotly
            fig = px.scatter(
                df_cluster, 
                x='Consumo_Energia_kWh', y='Desperdicio_Material_%', 
                color='Grupo_IA', # Pinta cada bolha conforme o grupo descoberto
                hover_name='ID_Obra', # Nome que aparece ao passar o mouse
                hover_data=['Horas_Atraso'], # Mostra dado extra no pop-up
                color_discrete_sequence=['#440154', '#21918c', '#fde725']
            )
            
            # Adiciona os X pretos marcando o "coração" (Centroide) de cada agrupamento
            fig.add_trace(
                go.Scatter(
                    x=centroides[:, 0], y=centroides[:, 1], mode='markers',
                    marker=dict(color='black', symbol='x', size=12, line=dict(width=2)),
                    name='Ponto Central'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("⚠️ Nota Técnica: Em produção, seria fundamental aplicar 'StandardScaler' antes do KMeans, pois a escala da Energia (milhares) é muito maior que o Desperdício (dezenas).")
            
    except FileNotFoundError:
        st.error("⚠️ Arquivo 'gestao_canteiros.csv' não encontrado.")

# ====================================================================
# MÓDULO 4: IA GENERATIVA (CHATBOT RAG)
# ====================================================================
elif menu == "💬 Assistente RAG (Manutenção)":
    st.title("💬 Assistente Virtual de Manutenção")
    st.markdown("Chatbot conectado ao Manual Técnico via RAG (Retrieval-Augmented Generation).")
    
    # 1. Painel de credenciais. O st.expander permite esconder essa caixa por segurança
    with st.expander("⚙️ Configurações da API Azure"):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("🔑 Chave da API", type="password") # type="password" oculta os caracteres digitados
        with col2:
            endpoint = st.text_input("🔗 Endpoint", value="https://marcelomaffeis-05082026-resource.services.ai.azure.com/openai/v1")
        modelo_azure = st.text_input("🧠 Nome do Deployment", value="gpt-4.1-mini")
    
    st.markdown("---")
    
    # 2. Injetando a Base de Conhecimento (Contexto)
    try:
        # Tenta abrir o arquivo TXT em modo leitura ('r') com codificação utf-8
        with open('manual_falhas_maquinas.txt', 'r', encoding='utf-8') as arquivo:
            base_conhecimento = arquivo.read()
    except FileNotFoundError:
        base_conhecimento = "ERRO: Arquivo não encontrado."
        st.error(base_conhecimento)

    # 3. Gerenciamento de Memória do Chat (Session State)
    # Se for a primeira vez abrindo a página, inicializamos a memória
    if "mensagens" not in st.session_state:
        # Instrução 'system' é a regra mestre que a IA deve obedecer, incluindo o manual técnico
        instrucao_sistema = f"""Você é um engenheiro sênior de manutenção. 
        Responda as dúvidas baseando-se EXCLUSIVAMENTE neste manual técnico abaixo.
        Se a resposta não estiver no manual, diga que precisa chamar o supervisor.
        
        MANUAL TÉCNICO:
        {base_conhecimento}
        """
        # Guardamos a regra na memória do Streamlit
        st.session_state.mensagens = [{"role": "system", "content": instrucao_sistema}]

    # 4. Desenhando o Chat na Tela
    # Um laço de repetição (for) para mostrar todo o histórico guardado na memória
    for msg in st.session_state.mensagens:
        if msg["role"] != "system": # Ignoramos o 'system' para não poluir a tela do usuário
            with st.chat_message(msg["role"]): # Cria o balãozinho (usuário ou assistente)
                st.markdown(msg["content"])

    # 5. Interação (A barra de digitação inferior)
    pergunta = st.chat_input("Ex: Minha grua está com alarme e bloqueada, o que eu faço?")
    
    if pergunta:
        if not api_key:
            st.warning("⚠️ Insira a Chave da API nas configurações acima para liberar o chat.")
        else:
            # Mostra a pergunta do usuário na tela e salva na memória
            with st.chat_message("user"):
                st.markdown(pergunta)
            st.session_state.mensagens.append({"role": "user", "content": pergunta})
            
            # O assistente começa a pensar
            with st.chat_message("assistant"):
                with st.spinner("Consultando o manual técnico..."): # Mostra o ícone de carregamento
                    try:
                        # 1. Abre a conexão segura com a Nuvem Azure
                        cliente = OpenAI(base_url=endpoint, api_key=api_key)
                        
                        # 2. Envia a memória inteira (regras + conversa) e pede uma resposta
                        resposta = cliente.chat.completions.create(
                            model=modelo_azure,
                            messages=st.session_state.mensagens
                        )
                        # 3. Extrai apenas o texto da resposta devolvida pela Nuvem
                        texto_ia = resposta.choices[0].message.content
                        
                        # Mostra a resposta e salva na memória
                        st.markdown(texto_ia)
                        st.session_state.mensagens.append({"role": "assistant", "content": texto_ia})
                        
                    except Exception as e:
                        st.error(f"Falha na comunicação com a IA: {e}")

