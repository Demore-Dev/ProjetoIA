import pandas as pd
import os  
import ofxparse
from datetime import datetime

# Cria um DataFrame que será usado para concatenar
# todos os DF temporários (gerados de cada extrato)
df = pd.DataFrame()

# Este bloco todo: abre cada extrato, pega os dados "Data", "Valor", "Descrição", "ID"
# e coloca em uma lista 'transaction_data', que vira um DataFrame temporário 
# (equivalente a cada extrato), e cada DF temporário é concatenado em 1 só depois

for extrato in os.listdir("extratos"): # Para cada arquivo na pasta extrato 
    with open(f'extratos/{extrato}') as ofx_file: # Abre cada arquivo 
        ofx = ofxparse.OfxParser.parse(ofx_file) # Processa o arquivo

    # Inicializa uma lista para armazenar os dados das transações
    transactions_data = []

    # Para cada transação, adiciona na lista os elementos especificados
    for account in ofx.accounts:
        for transaction in account.statement.transactions:
            transactions_data.append({
                "Data": transaction.date,
                "Valor": transaction.amount,
                "Descrição": transaction.memo,
                "ID": transaction.id,
            })

    # Converte a lista de transações em um DF temporário 
    # (gera um DF novo para cada arquivo extrato na pasta)
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
# LLM's
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

#Encontrando a chave API
_ = load_dotenv(find_dotenv('key.env'))

# Prompt que faz a classificação:
template = """
Você é um analista de dados, trabalhando em um projeto de limpeza de dados. Seu trabalho é escolher uma categoria adequada para cada lançamento financeiro que vou te enviar. Todos são transações financeiras de uma pessoa física. 
Nunca coloque nenhum tipo de pontuação nas categorias, como vírgulas ',' e pontos '.'

Escolha uma dentre as seguintes categorias:
- Alimentação
- Saúde
- Mercado
- Educação
- Compras pessoais
- Transporte
- Transferência para terceiros
- Telefone / Internet
- Moradia

Dicas: Sempre que você julgar que o item parece um nome de pessoa, minha sugestão é que escolha a categoria "Transferência para terceiros"

Escolha a categoria deste item:
{text}

Responda apenas com a categoria.
"""

# Cria um template de prompt usando a classe PromptTemplate,
# com o template que acabei de escrever
prompt = PromptTemplate.from_template(template=template)

# Inicializa o modelo de IA que eu quero utilizar (neste caso, o llama)
chat = ChatGroq(model="llama-3.2-90b-text-preview")

# Combina o prompt e o modelo de LLM para formar uma "chain" (cadeia) de operações,
# na qual envia o prompt ao modelo, e recebe uma resposta
chain = prompt | chat

# Bloco responsável por enviar ao modelo cada uma das descrições de compra
# do meu DF, e armazenar a resposta no vetor 'category', posteriormente
# transformando esse vetor em uma nova coluna "Categoria" no meu DF
category = [] 
for transaction in list(df["Descrição"].values):  
    response = [chain.invoke(transaction).content] # Recebe a resposta do modelo
    category.append(response) # Adiciona a resposta no vetor

# Cria a nova coluna, utilizando os elementos de 'category'
df["Categoria"] = category
# Exportando o DF como .csv, para fazer dashboard
df.to_csv("Planilha.csv")

# Aqui termina toda a parte de classificação

#===============================================================================================


