import streamlit as st
import pandas as pd
import ofxparse
import plotly.express as px
from io import StringIO
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

st.set_page_config(layout="wide")  # Configuração de layout

# Janela que solicita o carregamento de arquivos do usuário
st.write("# Por favor, carregue um ou mais arquivos '.ofx' para visualizar o painel")
uploaded_files = st.file_uploader("", type="ofx", accept_multiple_files=True)

# Inicializa um DataFrame vazio para armazenar os dados
df = pd.DataFrame()

# Se houver arquivos carregados
if uploaded_files:
    for uploaded_file in uploaded_files: # Para cada Arquivo carregado
        with StringIO(uploaded_file.getvalue().decode("utf-8")) as ofx_file:
            ofx = ofxparse.OfxParser.parse(ofx_file) # Processa cada arquivo
            
            # Inicializa uma lista para armazenar os dados das transações
            transactions_data = []

            # Para cada transação, adiciona na lista os elementos especificados
            for account in ofx.accounts:
                for transaction in account.statement.transactions:
                    transactions_data.append({
                        "Data": transaction.date,
                        "Valor": transaction.amount,
                        "Descrição": transaction.memo,
                    })

            # Converte a lista de transações em um DF temporário 
            # (gera um DF novo para cada arquivo)
            df_temp = pd.DataFrame(transactions_data)
            
            # Trata algumas informações
            df_temp["Valor"] = df_temp["Valor"].astype(float) # Convertendo o valor pra float
            df_temp["Data"] = df_temp["Data"].apply(lambda x: x.date()) # Tirando o horário
            
            df = pd.concat([df, df_temp]) # Adiciona o DF desse extrato no DF geral
    
    # Filtrar apenas os gastos (valores negativos)
    df = df[df["Valor"] < 0]

    # Aqui termina o processo de transformar cada arquivo em só 1 DF,
    # já pronto e com as informações que eu quero
    #===============================================================================================
    # LLM

    #Encontrando a chave API
    _ = load_dotenv(find_dotenv('key.env'))
    # Prompt que faz a classificação:
    template = """
    Você é um analista de dados, trabalhando em um projeto de limpeza de dados. Seu trabalho é escolher uma categoria adequada para cada lançamento financeiro que vou te enviar. UTILIZE APENAS AS CATEGORIAS PREDEFINIDAS. JAMAIS DÊ EXPLICAÇÕES. Caso encontre alguma transação e não consiga categorizá-la, escolha "Outros". Quando o item for um nome COMPLETO, escolha "Transferência para terceiros"

    Categorias:
    - Alimentação
    - Saúde
    - Educação
    - Lazer
    - Transporte
    - Transferência para terceiros
    - Internet
    - Moradia
    - Outros

    Escolha uma das categorias acima para este item: {text} (Responda APENAS com a categoria)
    """

    # Cria um template de prompt usando a classe PromptTemplate,
    # com o template que acabei de escrever
    prompt = PromptTemplate.from_template(template=template)

    # Inicializa o modelo de IA que eu quero utilizar (neste caso, o llama)
    chat = ChatGroq(model="llama-3.1-70b-versatile")

    # Combina o prompt e o modelo de LLM para formar uma "chain" (cadeia) de operações,
    # na qual envia o prompt ao modelo, e recebe uma resposta
    chain = prompt | chat

    # Bloco responsável por enviar ao modelo cada uma das descrições de compra
    # do meu DF, e armazenar a resposta no vetor 'category', posteriormente
    # transformando esse vetor em uma nova coluna "Categoria" no meu DF
    category = [] 
    for transaction in list(df["Descrição"].values):  
        response = chain.invoke(transaction).content # Recebe a resposta do modelo
        category.append(response) # Adiciona a resposta no vetor

    # Cria a nova coluna, utilizando os elementos de 'category'
    df["Categoria"] = category
    
    # Aqui termina a parte de classificação, que utiliza uma LLM para atribuir categorias às transações
    #====================================================================================================================
    #Aqui começa a parte de plotar o gráfico e criar o 'site'
    # Definindo uma cor para cada categoria, através de um dicionário
    cores_categoria = {
        "Alimentação": "green",
        "Saúde": "blue",
        "Mercado": "yellow",
        "Educação": "orange",
        "Compras pessoais": "pink",
        "Transporte": "red",
        "Transferência para terceiros": "purple",
        "Telefone / Internet": "brown",
        "Moradia": "grey",
        "Outros": "white"
    }    

    @st._cache_data
    def filtrar_dados(df, meses_selecionados, categorias_selecionadas):
        df_filtrado = df[df["Mês"].isin(meses_selecionados)]
        if categorias_selecionadas:
            df_filtrado = df_filtrado[df_filtrado["Categoria"].isin(categorias_selecionadas)]
        return df_filtrado

    # Bloco que realiza uma limpeza e algumas classificações:
    # Convertendo a coluna "Data" para datetime, e criando uma nova coluna "Data Original"
    df["Data original"] = pd.to_datetime(df["Data"], format="%Y-%m-%d", errors='coerce')  
    df["Mês"] = df["Data original"].dt.strftime("%m/%y")    # Extraindo o mês e ano no formato MM/YYYY
                                                            # (também crio uma nova coluna 'mês')
    df["Receita"] = df["Valor"] # Criando uma nova coluna com os valores ainda negativos
    df["Valor"] = df["Valor"].abs()  # Transformando os valores em positivos (absolutos)
    df["Categoria"] = df["Categoria"].str.replace(r"[\'\[\]\.\"]", "", regex=True)  # Removendo possíveis caracteres 

    # Interface
    st.title("Facilitador de Controle Financeiro")

    meses_selecionados = st.sidebar.multiselect("Selecione as datas:", df["Mês"].unique(), default=df["Mês"].unique())
    categorias = df["Categoria"].unique().tolist()
    categorias_selecionadas = st.sidebar.multiselect("Selecione as categorias:", categorias, default=categorias)
    df_filtrado = filtrar_dados(df, meses_selecionados, categorias_selecionadas)
    
    


    c1, c2 = st.columns([0.5, 0.5])

    # Formatando a data para exibir ao usuário (criando uma nova coluna formatada)
    df_filtrado["Data"] = df_filtrado["Data original"].dt.strftime("%d/%m/%Y")

    c1.dataframe(df_filtrado[["Data", "Descrição", "Receita", "Categoria"]])

    # Escolhendo o gráfico de 'pie' e definindo o que será exibido
    fig = px.pie(df_filtrado, title='Distribuição por Categoria', values="Valor", names="Categoria", color="Categoria", color_discrete_map=cores_categoria, hole=0.3)
    # Adicionar o valor junto com a porcentagem
    fig.update_traces(textinfo='value+percent', texttemplate='<br>R$%{value:,.2f}<br>(%{percent})')

    c2.plotly_chart(fig, use_container_width=True)