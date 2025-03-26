import streamlit as st
import pandas as pd
import json
import time
from workflow import run_fund_selection_workflow, load_data
import subprocess
from get_record_info import get_record_id_from_name
from langchain_openai import ChatOpenAI
from typing import TypedDict, Optional, Dict
from langchain.output_parsers.structured import StructuredOutputParser
import asyncio
from pydantic import BaseModel
from services.web_scraper import get_search_results
import os

# Carregar dados do DataFrame
df = load_data()

os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws"]["access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws"]["secret_access_key"]
os.environ["AWS_REGION"] = st.secrets["aws"]["region"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["anthropic"]["api_key"]
os.environ["SUPABASE_URL"] = st.secrets["supabase"]["url"]
os.environ["SUPABASE_KEY"] = st.secrets["supabase"]["key"]

# Configuração da página
st.set_page_config(
    page_title="Walter Intro Maker",
    page_icon="🤖",
    layout="wide"
)

# Título principal
st.title("Walter Intro Maker")
st.subheader("Intros to Funds Matcher")

# Criando abas
tab1, tab2 = st.tabs(["Search Information", "Parameters"])

# Inicializar estado da sessão para parâmetros e resultados
if 'parameters' not in st.session_state:
    st.session_state.parameters = {
        "batch_size": 10,
        "surviving_percentage": 1,
        "gdoc_id": "1AkNbFeXe5dvuzBVhFQUDfPh7B51YmjhasSGRUW4mMm0",
        "use_docs": False
    }

if 'results' not in st.session_state:
    st.session_state.results = None

if 'inputs' not in st.session_state:
    st.session_state.inputs = None

if 'progress' not in st.session_state:
    st.session_state.progress = None

if 'company_data' not in st.session_state:
    st.session_state.company_data = {
        "company": "",
        "description_company": "",
        "description_person": "",
        "industry": "",
        "round_size": 10,
        "round_type": "Series A",
        "round_commitment": 2,
        "leader_or_follower": "leader",
        "fund_closeness": "Close",
        "fund_quality": "High",
        "observations": "",
    }

# Definir a estrutura de saída para o LLM
class Round(TypedDict):
    round_size: float
    round_type: str
    round_commitment: float

class CompanyInfo(TypedDict):
    """
    Informações estruturadas sobre uma empresa.
    """
    description_company: str
    description_person: Optional[str]
    round: Round
    industry: str
    observations: str

# Função para extrair informações da empresa usando LLM
def extract_company_info(company_record):

    print(f"Company record: {company_record}")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Configurar LLM para retornar saída estruturada
    llm = llm.with_structured_output(CompanyInfo)
    
    prompt = f"""
    Baseado nas informações da empresa a seguir, extraia dados para preencher um formulário.
    
    Informações da empresa:
    {company_record}
    
    Por favor, forneça as seguintes informações (se disponíveis):
    1. Uma descrição completa da empresa (description_company)
    2. A indústria/setor de atuação (industry) - forneça múltiplos setores separados por vírgulas
    3. Informações sobre a rodada de investimento:
       - Tamanho aproximado em milhões de USD (round_size)
       - Tipo de rodada (round_type) por exemplo: Seed, Series A, etc.
       - Quanto já foi investido (round_commitment). In doubt, make it 0.
    4. Descrição do representante ou CEO (description_person)
    5. Any other relevant information (observations). Try to include things like the company's website, employee count, list  earances... any information.
    
    Se alguma informação não estiver disponível, coloque ["NOT FOUND"]
    """
    
    # O LLM já retornará diretamente a estrutura definida em CompanyInfo
    try:
        company_data = llm.invoke(prompt)

        print(f"Company data: {company_data}")
                
        # Garantir que campos numéricos sejam do tipo correto
        if "round_size" in company_data and company_data["round_size"]:
            try:
                company_data["round_size"] = float(company_data["round_size"])
                company_data["round_commitment"] = float(company_data["round_commitment"])
            except (ValueError, TypeError):
                company_data["round_size"] = 10  # valor padrão
                company_data["round_commitment"] = 0
                
        return company_data
    except Exception as e:
        st.error(f"Erro ao processar dados da empresa: {str(e)}")
        return {
            "description_company": "",
            "description_person": "",
            "industry": "",
            "round": {"round_size": 10, "round_type": "", "round_commitment": 0},
            "observations": f"Erro ao processar: {str(e)}"
        }

# Adicionar após a definição da classe CompanyInfo
async def enrich_company_information(company_name: str, industry: str) -> dict:

    class Query(BaseModel):
        query_name: str
        query_market: str

    # Criar queries para busca
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    query_prompt = f"""
    Crie duas queries de busca diferentes para obter informações sobre:
    1. Uma query que me dá informações sobre {company_name}
    2. Uma query que me dá estatísticas quantitativas sobre o mercado {industry}
    
    Retorne apenas as duas queries, uma por linha, sem numeração ou texto adicional.
    """
    
    queries = llm.with_structured_output(Query).invoke(query_prompt)
    company_query, market_query = queries.query_name, queries.query_market
    
    # Realizar web scraping
    company_results = await get_search_results(company_query, max_results=2)
    market_results = await get_search_results(market_query, max_results=4)
    
    # Consolidar informações usando LLM
    consolidation_prompt = f"""
    Analise as informações coletadas e crie um resumo estruturado.
    
    Informações da empresa:
    {json.dumps(company_results, indent=2)}
    
    Informações do mercado:
    {json.dumps(market_results, indent=2)}
    
    Formate o resumo em tópicos separados para Empresa e Mercado.
    """
    
    summary = llm.invoke(consolidation_prompt)
    
    return {
        "company_info": company_results,
        "market_info": market_results,
        "summary": summary
    }

with tab1:
    # Campo para buscar empresa por nome
    company_name = st.text_input("Buscar empresa por nome", value=st.session_state.company_data["company"])

    # Adicionar seção de informações enriquecidas
    check = st.checkbox("Enrich with web search (takes more time, not performing well yet)")
    
    if st.button("Buscar informações"):
        try:
            with st.spinner("Buscando informações da empresa..."):
                # Obter informações da empresa usando a função get_record_id_from_name
                company_record = get_record_id_from_name(company_name, "companies")
                
                # Extrair informações relevantes usando LLM
                company_info = extract_company_info(company_record)
                
                # Atualizar o estado da sessão com as informações obtidas
                st.session_state.company_data.update({
                    "company": company_name,
                    **company_info
                })
            if check:
                with st.expander("Informações Adicionais"):
                    with st.spinner("Buscando informações complementares..."):
                        enriched_info = asyncio.run(enrich_company_information(company_name, company_info))
                        
                        # Exibir resultados
                        st.text("Informações coletadas da web:")
                        st.markdown(enriched_info["summary"].content)
                        
                        st.write("Fontes sobre a empresa:")
                        for result in enriched_info["company_info"]:
                            st.write(f"- [{result.get('title', 'Link')}]({result.get('url', '#')})")
                    
                        st.write("Fontes sobre o mercado:")
                        for result in enriched_info["market_info"]:
                            st.write(f"- [{result.get('title', 'Link')}]({result.get('url', '#')})")
                        
                        

                        st.success(f"Informações de {company_name} encontradas e preenchidas!")
        except Exception as e:
            st.error(f"Erro ao buscar informações: {str(e)}")

    # Formulário para dados da empresa
    with st.form("company_form"):
        st.subheader("Company Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Company Information")
            company = st.text_input("Company Name", value=st.session_state.company_data["company"])
            description_company = st.text_area(
                "Company Description", 
                value=st.session_state.company_data["description_company"]
            )
            description_person = st.text_area(
                "Representative Description", 
                value=st.session_state.company_data["description_person"]
            )
            industry = st.text_input(
                "Industry", 
                value=st.session_state.company_data["industry"]
            )
        
        with col2:
            st.text("Fundraising Information")
            round_size = st.number_input("Round Size (in millions of USD)", value=st.session_state.company_data["round_size"], help="Tamanho da rodada em milhões de USD")
            round_type = st.text_input("Funding Type", value=st.session_state.company_data["round_type"])
            round_commitment = st.number_input("Round Commitment (in millions of USD)", value=st.session_state.company_data["round_commitment"], help="Tamanho da rodada em milhões de USD")
            leader_or_follower = st.selectbox(
                "Position in Round (Are we looking for a leader or a follower?)",
                options=["leader", "follower", "both"],
                index=["leader", "follower", "both"].index(st.session_state.company_data["leader_or_follower"])
            )
            fund_closeness = st.selectbox(
                "Fund Proximity (How close we want the fund to be to us?)",
                options=["Close", "Distant", "Irrelevant"],
                index=["Close", "Distant", "Irrelevant"].index(st.session_state.company_data["fund_closeness"])
            )

            fund_quality = st.selectbox(
                "Fund Quality",
                options=["High", "Medium", "Low"],
                index=["High", "Medium", "Low"].index(st.session_state.company_data["fund_quality"])
            )
            
        observations = st.text_area(
            "Additional Observations", 
            value=st.session_state.company_data["observations"]
        )
        
        submitted = st.form_submit_button("Run Recommendations")
        
        if submitted:
            # Atualizar os dados da empresa no estado da sessão
            st.session_state.company_data = {
                "company": company,
                "description_company": description_company,
                "description_person": description_person,
                "industry": industry,
                "round_size": round_size,
                "round_type": round_type,
                "round_commitment": round_commitment,
                "leader_or_follower": leader_or_follower,
                "fund_closeness": fund_closeness,
                "fund_quality": fund_quality,
                "observations": observations
            }
            
            # Criar o dicionário de inputs
            st.session_state.inputs = {
                "company": company,
                "description_company": description_company,
                "description_person": description_person,
                "round": {"size": round_size, "Funding": round_type},
                "round_commitment": round_commitment,
                "leader_or_follower": leader_or_follower,
                "industry": industry,
                "fund_closeness": fund_closeness,
                "fund_quality": fund_quality,
                "observations": observations
            }
            
            # Iniciar processamento em segundo plano e mudar para a aba de resultados
            st.session_state.progress = "starting"
            st.rerun()


with tab2:
    st.subheader("Generation Parameters")
    
    # Formulário para parâmetros
    with st.form("parameters_form"):
        batch_size = st.slider("Batch Size", 1, 50, int(st.session_state.parameters.get("batch_size", 10)))
        surviving_percentage = st.slider("Survival Percentage", 0.1, 1.0, float(st.session_state.parameters.get("surviving_percentage", 1)), 0.1)
        
        # Adicionar campo para ID do Google Doc
        gdoc_id = st.text_input("ID do Google Doc", 
                               value=st.session_state.parameters.get("gdoc_id", "1AkNbFeXe5dvuzBVhFQUDfPh7B51YmjhasSGRUW4mMm0"),
                               help="ID do documento do Google que contém informações adicionais")
        
        params_submitted = st.form_submit_button("Save Parameters")

        use_docs = st.checkbox("Use Google Docs", value=st.session_state.parameters.get("use_docs", False))
        
        if params_submitted:
            st.session_state.parameters = {
                "batch_size": batch_size,
                "surviving_percentage": surviving_percentage,
                "gdoc_id": gdoc_id,  # Adicionar o ID do Google Doc aos parâmetros
                "use_docs": use_docs
            }
            
            st.success("Parâmetros salvos com sucesso!")

st.subheader("Analysis Results")

st.info("This demo takes a while to run since it runs fund by fund. Please be patient.")

# Verificar se o processamento deve começar
if st.session_state.progress == "starting" and st.session_state.inputs:
    # Container para exibir progresso
    progress_container = st.empty()
    status_container = st.empty()
    
    # Mostrar processo de execução
    try:

        # Carregar dados
        status_container.info("Loading data...")
        progress_container.progress(10)
        
        # Processar seleção de fundos
        status_container.info("Analyzing compatible funds...")
        progress_container.progress(30)
        
        # Chamar a função do workflow
        results = run_fund_selection_workflow(
            st.session_state.inputs, 
            st.session_state.parameters
        )
        
        progress_container.progress(100)
        status_container.success("Processing completed!")
        
        # Armazenar resultados
        st.session_state.results = results
        st.session_state.progress = "completed"
        
        # Recarregar para exibir resultados completos
        st.rerun()
        
    except Exception as e:
        status_container.error(f"Error during processing: {str(e)}")
        st.session_state.progress = None

# Exibir resultados se disponíveis
if st.session_state.results:
    # Exibir tabela com os melhores fundos
    st.subheader("Selected Funds")
    
    # Criar DataFrame para exibição
    fund_data = []
    for fund in st.session_state.results["top_funds"]:
        cols_to_show = ["proximity", "Fund Quality","description", "leader?", "prefered_industry_enriched"]
        df.rename(columns={"vc_quality_perception": "Fund Quality"}, inplace=True)
        # Buscar todas as informações do fundo no DataFrame original
        fund_info = df[cols_to_show][df["name"] == fund.fund_name].iloc[0].to_dict()
        fund_data.append({
            "Fund Name": fund.fund_name,
            "Score": round(fund.score, 0),
            "Reason": fund.reason,
            **fund_info  # Incluir todas as outras colunas do fundo
        })
    
    result_df = pd.DataFrame(fund_data)
    st.dataframe(result_df)
    
    # Botão para baixar resultados como CSV
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="selected_funds.csv",
        mime="text/csv"
    )

elif st.session_state.progress is None:
    st.info("Fill in the company information and click 'Generate Introduction' to analyze compatible funds.")

# Rodapé
st.markdown("---")
st.markdown("Developed by Norte Ventures")
