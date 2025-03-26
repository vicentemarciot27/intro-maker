import os
from dotenv import load_dotenv
import operator
import logging
import pandas as pd
import math
from typing import List, Dict, Any
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from google.oauth2 import service_account
import gspread
from googleapiclient.discovery import build
import concurrent.futures
from functools import partial
import boto3
from botocore.config import Config
import streamlit as st

service_account_info = st.secrets["service_account"]

config = Config(read_timeout=1000)

client = boto3.client(service_name='bedrock-runtime', 
                      region_name='us-east-1',
                      config=config)

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do modelo Claude
def configure_claude():
    return ChatBedrock(
        model_id="arn:aws:bedrock:us-east-1:050451404360:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        provider="anthropic",
        model_kwargs={"max_tokens": 20000},
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        client=client
    )

def configure_haiku():
    return ChatBedrock(
        model_id="arn:aws:bedrock:us-east-1:050451404360:inference-profile/us.anthropic.claude-3-haiku-20240307-v1:0",
        provider="anthropic",
        model_kwargs={"max_tokens": 20000},
    )

def configure_o3():
    return ChatOpenAI(
        model="o3-mini"
    )

def configure_gpt_4o_mini():
    return ChatOpenAI(
        model="gpt-4o-mini"
    )

# Classes para estruturar os resultados
class FundScore(BaseModel):
    fund_name: str = Field(description="Fund Name")
    score: float = Field(description="Gross Score based on the sum of the criteria")
    reason: str = Field(description="Detailed reason for the score separated by criteria")

class FundScoreList(BaseModel):
    scores: List[FundScore]

# Carregamento e preparação dos dados
def load_data():
    # Configurar credenciais
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/spreadsheets', 
                'https://www.googleapis.com/auth/drive']
    )
    
    # Conectar ao Google Sheets
    gc = gspread.authorize(credentials)
    
    # Extrair o ID da planilha
    sheet_id = "11I9QFSMFn7UBfV0wz0-hAYgWtIKytTVnWA9pjquwgdk"
    
    # Abrir a planilha e obter a primeira aba
    sheet = gc.open_by_key(sheet_id).sheet1
    
    # Obter todos os dados e converter para DataFrame
    data = sheet.get_all_values()
    headers = data[0]
    df = pd.DataFrame(data[1:], columns=headers)
    
    return df

# Carregamento de documentos do Google Docs
def setup_gdocs():
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )
    
    # Criar serviço do Google Docs
    return build('docs', 'v1', credentials=creds)

def get_gdoc_content(docs_service, doc_id):
    try:
        document = docs_service.documents().get(documentId=doc_id).execute()
        
        # Extrair texto do documento
        doc_plaintext = ""
        for item in document.get('body', {}).get('content', []):
            if 'paragraph' in item:
                for element in item.get('paragraph', {}).get('elements', []):
                    if 'textRun' in element:
                        doc_plaintext += element.get('textRun', {}).get('content', '')
        
        return {
            'title': document.get('title', ''),
            'content': doc_plaintext
        }
    except Exception as e:
        print(f"Erro ao acessar o documento: {e}")
        return None

# Filtragem inicial dos dados
def filter_data(df, inputs):
    # Tratar valores vazios nas colunas numéricas
    df["vc_quality_perception"] = df["vc_quality_perception"].replace("", 0)
    df["vc_quality_perception"] = df["vc_quality_perception"].astype(float, errors='ignore')
    df["proximity"] = df["proximity"].replace("", 0)
    df["proximity"] = df["proximity"].astype(float, errors='ignore')
    
    # Determinar o range de investimento com base no tamanho da rodada
    round_size = float(str(inputs["round"]["size"]).replace("M USD", "").strip())
    
    if round_size < 1:
        company_investment_range = ["< USD 1mn"]
    elif round_size < 5:
        company_investment_range = ["USD 5-10mn", "< USD 1mn"]
    elif round_size < 10:
        company_investment_range = ["USD 10-20mn", "USD 5-10mn", "< USD 1mn"]
    else:
        company_investment_range = [">USD 20mn", "USD 10-20mn", "USD 5-10mn", "< USD 1mn"]
    
    # Criar padrão para filtrar investment_range
    df["investment_range"] = df["investment_range"].str.strip("[]")
    company_investment_range_pattern = '|'.join(company_investment_range)
    
    # Aplicar filtros
    df = df[df["investment_range"].str.contains(company_investment_range_pattern, na=False)]
    
    # Filtrar por leader/follower
    if inputs["leader_or_follower"] == "leader":
        df = df[df["leader?"].str.lower().str.contains("leader")]
    elif inputs["leader_or_follower"] == "follower":
        df = df[df["leader?"].str.lower().str.contains("follower")]
    
    # Filtrar por qualidade do fundo, se presente nos inputs
    if "fund_quality" in inputs:
        if inputs["fund_quality"] == "High":
            df = df[df["vc_quality_perception"] >= 4]
        elif inputs["fund_quality"] == "Medium":
            df = df[df["vc_quality_perception"] >= 3]
        elif inputs["fund_quality"] == "Low":
            df = df[df["vc_quality_perception"] < 3]
    

    # Filtrar por proximidade do fundo, se presente nos inputs
    if "fund_closeness" in inputs and inputs["fund_closeness"] == "Close":
        df = df[df["proximity"] >= 3]
    elif inputs["fund_closeness"] == "Distant":
        df = df[df["proximity"] <= 3]
    

    return df

# Dividir em lotes
def batch_splitter(df, batch_size):
    return [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

# Função para processar um único lote
def process_batch(batch, inputs, parameters, llm, previous_scores=None, gdoc_content=None, batch_index=0, total_batches=0):
    use_docs = parameters.get("use_docs", False)
    # Preparar orientação baseada em pontuações anteriores
    previous_scores_guidance = ""
    if previous_scores:
        # Criar exemplos de pontuações anteriores para manter consistência
        examples = [(s.fund_name, s.score) for s in previous_scores[:5]]
        previous_scores_guidance = f""" 
        IMPORTANT: Keep consistency with the scores already assigned to other funds.
        Examples of previous scores: {examples}
        Remember that the total score must be in an approximate scale with the scores already assigned.
        """
    
    system_prompt = """
    You are a fund score agent. Score every fund.
    You are given a table of funds, user inputs, and you need to score them based on the following criteria:

    - preffered_industry (the fund's preferred industry should be compatible with the company's industry. If there is just one intersection, the score is around 5. If there is a near perfect fit, the score is 10) | 0-10 points
    - investment_geography (the fund's investment geography should be compatible with the user's investment geography) | -5 to 5 points. If the geography is a perfect match, the score is 5. If the geography is not a perfect match, the score is 3. If incompatible, the score is -5
    - funding_rounds_1st_check (the first check round should be compatible with the round type) | 0-5 points
    - description (the description should be compatible with the company's description) | 0-3 points
    - observations (Use it as a situational reference of the fund) | -5 to 5 points

    Begin "reason" with a summary of the decision. Don't use words like "perfect" and be objective. End the reason with observations about the decision.
    """
    
    # Adicionar critério para o conteúdo do Google Doc se disponível
    if gdoc_content:
        system_prompt += """
    - Google Doc Content (additional context from the provided document that should be considered when scoring the funds) | 0-15 points. Consider how well the fund aligns with the specific details provided in the document.
        """
    
    system_prompt += """
    {previous_scores_guidance}
    """

    human_prompt = """
    Here is the table of funds:
    {df}

    Here is the user inputs:
    DISCLAIMER: fund_closeness means we want a close fund. If fund_closeness is Distant, we want a distant fund.
    Inputs explaination:
    - company: the company we are looking for a fund to invest in
    - description_company: the description of the company we are looking for a fund to invest in
    - description_person: the description of the person we are looking for a fund to invest in
    - round: The round size
    - round_commitment: What is already taken in the round. We are looking for someone to invest in the range round_commitment - round.
    - leader_or_follower: If we want a leader, we should put leader. If we want a follower, we should put follower.
    - industry: the industry we are looking for a fund to invest in
    - fund_closeness: How close we want the funds to be to us, Norte.
    - observations: Any other information that should be considered when scoring the funds
    
    {inputs}

    Remember to score the funds based on the criteria and the inputs.
    """
    
    # Adicionar conteúdo do Google Doc ao prompt se disponível
    if gdoc_content:
        human_prompt += """
    
    Additional context from Google Doc "{title}":
    {content}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(previous_scores_guidance=previous_scores_guidance)),
        ("human", human_prompt)
    ])

    # Preparar variáveis para invocação
    variables = {"df": batch.to_string(), "inputs": inputs}
    
    # Adicionar conteúdo do Google Doc se disponível
    if gdoc_content and use_docs:
        variables["title"] = gdoc_content["title"]
        variables["content"] = gdoc_content["content"]
    
    structured_llm = llm.with_structured_output(FundScoreList)
    chain = prompt | structured_llm
    
    # Invocar o modelo
    try:
        fund_scores = chain.invoke(variables)
        print(f"Processado lote {batch_index+1}/{total_batches}")
        return fund_scores.scores
    except Exception as e:
        print(f"Erro ao processar lote {batch_index+1}: {str(e)}")
        return []

# Pontuação dos fundos com paralelização
def score_fund(df, inputs, parameters, model="claude"):
    cols_for_ai = ["name", "investment_geography", "prefered_industry_enriched", "description", "observations"]
    df = df[cols_for_ai]
    if model == "claude":
        llm_factory = configure_claude
    elif model == "o3":
        llm_factory = configure_o3
    elif model == "gpt-4o-mini":
        llm_factory = configure_gpt_4o_mini
    elif model == "haiku":
        llm_factory = configure_haiku

    # Verificar se um ID de Google Doc foi fornecido
    gdoc_content = None
    if parameters.get("gdoc_id") and parameters.get("use_docs"):
        try:
            docs_service = setup_gdocs()
            gdoc_content = get_gdoc_content(docs_service, parameters["gdoc_id"])
            print(f"Conteúdo do Google Doc carregado: {gdoc_content['title']}")
        except Exception as e:
            print(f"Erro ao carregar o Google Doc: {e}")
    
    batch_size = parameters.get("batch_size", 10)
    batches = batch_splitter(df, batch_size)
    
    raw_scores = []
    
    # Número máximo de worker threads
    max_workers = min(parameters.get("max_workers", 4), len(batches))
    
    print(f"Iniciando processamento paralelo com {max_workers} workers para {len(batches)} lotes")
    
    # Fase 1: Processar primeiro lote para obter pontuações de referência
    if batches:
        llm = llm_factory()
        first_batch_scores = process_batch(
            batches[0], 
            inputs, 
            parameters, 
            llm, 
            previous_scores=None, 
            gdoc_content=gdoc_content,
            batch_index=0,
            total_batches=len(batches)
        )
        raw_scores.extend(first_batch_scores)
        
        # Fase 2: Processar lotes restantes em paralelo
        remaining_batches = batches[1:]
        
        if remaining_batches:
            # Executar processamento paralelo
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Criar um LLM para cada worker
                futures = []
                for i, batch in enumerate(remaining_batches):
                    # Criar uma nova instância do modelo para cada worker
                    worker_llm = llm_factory()
                    
                    # Chamar diretamente a função sem usar partial
                    # Isso evita a confusão de argumentos que estava ocorrendo
                    futures.append(
                        executor.submit(
                            process_batch,
                            batch=batch,
                            inputs=inputs,
                            parameters=parameters,
                            llm=worker_llm,
                            previous_scores=raw_scores,
                            gdoc_content=gdoc_content,
                            batch_index=i+1,
                            total_batches=len(batches)
                        )
                    )
                
                # Coletar resultados à medida que são concluídos
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_scores = future.result()
                        raw_scores.extend(batch_scores)
                    except Exception as e:
                        print(f"Erro em worker thread: {str(e)}")

    return raw_scores

# Normalizar pontuações
def normalize_scores(raw_scores):
    if not raw_scores:
        return []
    
    min_score = min(score.score for score in raw_scores)
    max_score = max(score.score for score in raw_scores)
    
    if max_score > min_score:  # evitar divisão por zero
        return [
            FundScore(
                fund_name=score.fund_name,
                score=100 * (score.score - min_score) / (max_score - min_score),
                reason=score.reason
            ) for score in raw_scores
        ]
    else:
        # Caso todas as pontuações sejam iguais
        return [
            FundScore(
                fund_name=score.fund_name,
                score=50.0,  # valor médio arbitrário
                reason=score.reason
            ) for score in raw_scores
        ]

# Selecionar os melhores fundos
def select_top_funds(normalized_scores, percentage):
    # Ordenar scores
    sorted_scores = sorted(normalized_scores, key=lambda x: x.score, reverse=True)
    
    # Calcular número de fundos a manter
    to_keep = math.ceil(percentage * len(sorted_scores))
    
    # Retornar os melhores fundos
    return sorted_scores[:to_keep]

# Função principal que orquestra todo o fluxo
def run_fund_selection_workflow(inputs, parameters):

    use_docs = parameters.get("use_docs", False)

    # Carregar dados
    print("Carregando dados...")
    df = load_data()
    
    # Filtrar dados
    print("Filtrando dados...")
    filtered_df = filter_data(df, inputs)
    
    # Carregar conteúdo do Google Doc se disponível
    if parameters.get("gdoc_id") and use_docs:
        print(f"Carregando conteúdo do Google Doc: {parameters['gdoc_id']}...")
        try:
            docs_service = setup_gdocs()
            gdoc_content = get_gdoc_content(docs_service, parameters["gdoc_id"])
            print(f"Google Doc carregado: {gdoc_content['title']}")
        except Exception as e:
            print(f"Erro ao carregar o Google Doc: {e}")
    
    # Definir número máximo de workers se não estiver nos parâmetros
    if "max_workers" not in parameters:
        parameters["max_workers"] = 4
    
    # Pontuar fundos
    print("Pontuando fundos...")
    raw_scores = score_fund(filtered_df, inputs, parameters, model="o3")
    
    # Normalizar pontuações
    print("Normalizando pontuações...")
    normalized_scores = normalize_scores(raw_scores)
    
    # Selecionar os melhores fundos
    print("Selecionando melhores fundos...")
    surviving_percentage = parameters.get("surviving_percentage", 0.5)
    top_funds = select_top_funds(normalized_scores, surviving_percentage)
    
    # Extrair nomes dos fundos selecionados
    fund_names = [fund.fund_name for fund in top_funds]
    
    return {
        "top_funds": top_funds,
        "fund_names": fund_names
    }

# Exemplo de uso
if __name__ == "__main__":
    inputs = {
        "company": "Brendi",
        "description_company": "Brendi is a company that creates AI agents to sell food in Brazilian restaurants via delivery. They are going to be the next ifood",
        "description_person": "Daniel is the CEO of Brendi. he studied at ITA, is very young and energetic",
        "round": {"size": 10, "Funding": "Series A"},
        "round_commitment": "2M USD",
        "leader_or_follower": "leader",
        "industry": "AI Solutions, Food Delivery, Restaurant Management, AI Agents, Embedded Finance",
        "fund_closeness": "Distant",
        "observations": "The deal is cold. We want bad funds for it",
        "fund_quality": "Any"
    }

    parameters = {
        "batch_size": 10,
        "surviving_percentage": 1,
        "use_docs": False
    }
    
    results = run_fund_selection_workflow(inputs, parameters)
    # save results to a txt file
    print(f"Results: {results}"),
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(str(results))