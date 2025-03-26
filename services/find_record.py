import requests
import os
import json
from pathlib import Path
from collections import defaultdict
import os
import streamlit as st

os.environ["ATTIO_API_KEY"] = st.secrets["attio"]["api_key"]
ATTIO_API_KEY = st.secrets["attio"]["api_key"]

def get_list_name_from_slug(list_slug:str):
    lists_json = json.load(open(Path(__file__).parent / "lists.json"))
    for list in lists_json["data"]:
        if list["api_slug"] == list_slug:
            return list["name"]
    raise ValueError(f"List slug {list_slug} not found")

PATH_TO_LISTS_JSON = Path("./lists.json")

def list_record_entries(record_id: str, object: str):
    url = f"https://api.attio.com/v2/objects/{object}/records/{record_id}/entries"

    headers = {
        "Authorization": f"Bearer {ATTIO_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers).json()["data"]
    except Exception as e:
        raise RuntimeError(f"Erro ao buscar entradas do registro: {e}")

    # Agrupar entradas por lista
    entries_by_list = defaultdict(list)
    
    for entry in response:
        # Manter o campo created_at para ordenação
        entries_by_list[entry["list_api_slug"]].append(entry)
    
    # Obter apenas as entradas mais recentes de cada lista
    latest_entries = []
    
    for list_slug, entries in entries_by_list.items():
        # Ordenar entradas por data de criação (mais recente primeiro)
        if entries and "created_at" in entries[0]:
            entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Obter apenas a entrada mais recente
        if entries:
            latest_entry = entries[0]
            latest_entry["name"] = get_list_name_from_slug(list_slug)
            
            # Remover campos desnecessários
            latest_entry.pop("created_at", None)
            latest_entry.pop("list_id", None)
            
            # Buscar detalhes adicionais para a entrada
            try:
                entry_details = get_entry_details(list_slug, latest_entry["entry_id"])
                # Mesclar os detalhes com a informação básica da entrada
                latest_entry["details"] = entry_details
            except Exception as e:
                latest_entry["details_error"] = str(e)
            
            latest_entries.append(latest_entry)

    if object == "companies":
        notes = get_notes(record_id)
        latest_entries.append(notes)
    
    return latest_entries

def get_entry_details(list_slug: str, entry_id: str):
    """
    Busca informações detalhadas sobre uma entrada específica em uma lista
    """
    url = f"https://api.attio.com/v2/lists/{list_slug}/entries/{entry_id}"
    
    headers = {
        "Authorization": f"Bearer {ATTIO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            return {"error": f"Status code: {response.status_code}", "message": response.text}
    except Exception as e:
        return {"error": str(e)}
    
def get_notes(record_id: str):
    url = f"https://api.attio.com/v2/notes"

    headers = {
        "Authorization": f"Bearer {ATTIO_API_KEY}",
        "Content-Type": "application/json"
    }

    params = {
        "parent_object": "companies",
        "parent_record_id": record_id
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        return response.json()["data"]
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    print(list_record_entries("00010e3d-74b6-4471-90c3-ba6a637f901f", "companies"))
