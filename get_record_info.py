from langchain.tools import tool
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.output_parsers.structured import StructuredOutputParser
from typing import TypedDict
import time
from database.engine import create_db
from services.find_record import list_record_entries
from services.web_scraper import get_search_results
import os
import streamlit as st

os.environ["ATTIO_API_KEY"] = st.secrets["attio"]["api_key"]

db, engine = create_db()

def get_record_id_from_name(name: str, object: Literal["companies", "people"], additional_info: str = ""):
    """
    Get the record id of an object (company or person) from its name and other additional information. It's useful to send ids to sql_editor if you need any new information about the object.

    Args:
        name: The name of the object
        object: The object to get the record id from

    Returns:
        The record id of the object

    Example 1:
        name = "Norte"
        object = "companies"

    Example 2:
        name = "Bhub"
        object = "companies"

    Example 3:
        name = "Jota"
        object = "people"

    Example 4:
        name = "Daniel Frageri"
        object = "people"
    """

    if object == "companies":
        sql_query_results = get_record_id_candidates_from_name_companies(name)
    elif object == "people":
        sql_query_results = get_record_id_candidates_from_name_people(name)

    llm_evaluation = evaluate_sql_query_results(sql_query_results, name, additional_info)

    found_record_id = llm_evaluation["record_id"]

    # list record entries
    record_entries = list_record_entries(found_record_id, object)

    # query_name = create_query_name(name, additional_info)
    # query_market = create_query_market(name, additional_info)


    # search_results_name = await get_search_results(query_name)
    # search_results_market = await get_search_results(query_market)

    # add it to the llm_evaluation
    output = {
        **llm_evaluation,
        "record_entries": record_entries,
        # "search_results_name": search_results_name,
        # "search_results_market": search_results_market
    }

    return output

    
    
def get_record_id_candidates_from_name_companies(name: str, limit: int = 50):
    """
    Get the record id of a company from its name.
    """
    # Usando LOWER() em ambos os lados para busca case-insensitive
    query = """
        SELECT * 
        FROM nekt_trusted.attio_records_companies 
        WHERE LOWER(name) LIKE LOWER(:search_pattern)
        ORDER BY
	    "last_interaction"."interacted_at" DESC
        LIMIT :limit
    """
    
    result = db.run(query, parameters={
        "search_pattern": f"%{name.strip()}%",
        "limit": limit
    })
    return result

def get_record_id_candidates_from_name_people(name: str, limit: int = 50):
    """
    Get the record id of a person from its name.
    """
    query = f"SELECT * FROM nekt_trusted.attio_records_people WHERE name LIKE '%{name}%' LIMIT {limit}"
    
    result = db.run(query)
    return result

def evaluate_sql_query_results(sql_query_results: list[str], name: str, additional_info: str):
    """
    Evaluate the sql query results and return the best match.
    """

    class llmResponse(TypedDict):
        """
        Response from the LLM.

        Args:
            record_id: the main objective of the query, the id that corresponds to the name and additional_info
            reason: the reason for the choice of the record_id
            other_columns: other columns of the query results that add any interesting information about the object
        """
        record_id: str
        reason: str
        other_columns: dict[str, str]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = llm.with_structured_output(llmResponse)

    prompt = f"""
    You are a helpful assistant that evaluates the sql query results and returns the best match according to the name
    {"and some additional information on the company/person" if additional_info else ""}.

    The sql query results are the following:
    {sql_query_results}

    The name you are looking for is: {name}
    The information that appears first in the sql query results is the most recent interaction. Use that to weight the results.
    {"The additional information is: {additional_info}" if additional_info else ""}
    """
    
    response = llm.invoke(prompt)

    return llmResponse(
        record_id=response["record_id"],
        reason=response["reason"],
        other_columns=response["other_columns"]
    )

def create_query_name(name: str, additional_info: str):
    """
    Create a query name from a name and additional information.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
    You are a helpful assistant that creates a query name from a name and additional information.
    The name is: {name}
    The additional information is: {additional_info}
    """
    response = llm.invoke(prompt)
    return response

def create_query_market(name: str, additional_info: str):
    """
    Create a query market from a name and additional information.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
    You are a helpful assistant that queries the market of a company based on the name and additional information.
    The name is: {name}
    The additional information is: {additional_info}
    """
    response = llm.invoke(prompt)
    return response



if __name__ == "__main__":
    company = "Brendi"
    time_start = time.time()
    print(get_record_id_from_name(company, "companies"))
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

