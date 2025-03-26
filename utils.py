import json
import os
from typing import List, Dict, Any

from pydantic import BaseModel, Field

# Modelos para uso nas funções utilitárias
class FundScore(BaseModel):
    fund_name: str
    score: float
    reason: str

class FundScoreList(BaseModel):
    scores: List[FundScore] = Field(default_factory=list)

# Função para salvar resultados em JSON
def save_fund_scores(fund_scores: List[FundScore], filename: str = "fund_scores.json"):
    """
    Salva a lista de pontuações de fundos em um arquivo JSON.
    
    Args:
        fund_scores: Lista de objetos FundScore
        filename: Nome do arquivo para salvar os resultados
    """
    # Converter para dicionários
    scores_dict = [score.dict() for score in fund_scores]
    
    # Salvar no arquivo
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, ensure_ascii=False, indent=4)
    
    print(f"Pontuações salvas em {filename}")

# Função para carregar resultados de JSON
def load_fund_scores(filename: str = "fund_scores.json") -> List[FundScore]:
    """
    Carrega a lista de pontuações de fundos de um arquivo JSON.
    
    Args:
        filename: Nome do arquivo para carregar os resultados
        
    Returns:
        Lista de objetos FundScore
    """
    if not os.path.exists(filename):
        return []
    
    # Carregar do arquivo
    with open(filename, "r", encoding="utf-8") as f:
        scores_dict = json.load(f)
    
    # Converter para objetos FundScore
    return [FundScore(**score) for score in scores_dict]

# Função para filtrar fundos por pontuação mínima
def filter_funds_by_score(fund_scores: List[FundScore], min_score: float = 50.0) -> List[FundScore]:
    """
    Filtra a lista de fundos para incluir apenas aqueles com pontuação acima do limiar.
    
    Args:
        fund_scores: Lista de objetos FundScore
        min_score: Pontuação mínima para incluir um fundo (padrão: 50.0)
        
    Returns:
        Lista filtrada de objetos FundScore
    """
    return [score for score in fund_scores if score.score >= min_score]

# Função para formatar os resultados para exibição
def format_results_for_display(fund_scores: List[FundScore], company_name: str, limit: int = 10) -> str:
    """
    Formata os resultados para exibição.
    
    Args:
        fund_scores: Lista de objetos FundScore
        company_name: Nome da empresa
        limit: Número máximo de fundos a incluir (padrão: 10)
        
    Returns:
        String formatada com os resultados
    """
    # Ordenar por pontuação (do maior para o menor)
    sorted_scores = sorted(fund_scores, key=lambda x: x.score, reverse=True)
    
    # Limitar ao número especificado
    top_funds = sorted_scores[:limit]
    
    # Formatar a lista
    fund_list = "\n".join([f"{i+1}. {fund.fund_name}: {fund.score:.1f} - {fund.reason}" 
                         for i, fund in enumerate(top_funds)])
    
    # Montar a mensagem completa
    message = f"""
    ## Top Fundos Recomendados para {company_name}
    
    {fund_list}
    """
    
    return message

# Função para converter batch de DataFrame para formato adequado
def format_batch_for_llm(batch):
    """
    Formata um batch de DataFrame para uso com o LLM.
    
    Args:
        batch: DataFrame com dados dos fundos
        
    Returns:
        String formatada para envio ao LLM
    """
    # Selecionar colunas mais relevantes para análise
    relevant_columns = [
        "name", "industry_agnostic", "leader?", "investment_geography", 
        "preferred_industry", "vc_quality_perception", "observations", 
        "investment_range", "prefered_industry_enriched", "description"
    ]
    
    # Filtrar colunas que existem no DataFrame
    cols_to_use = [col for col in relevant_columns if col in batch.columns]
    
    # Criar uma representação textual mais limpa
    rows = []
    for _, row in batch.iterrows():
        fund_info = [f"Fund: {row.get('name', 'Unknown')}"]
        
        for col in cols_to_use:
            if col != 'name' and not pd.isna(row.get(col, None)):
                # Formatar o nome da coluna para ser mais legível
                col_name = col.replace('_', ' ').title()
                fund_info.append(f"{col_name}: {row[col]}")
        
        rows.append("\n".join(fund_info))
    
    return "\n\n".join(rows)
