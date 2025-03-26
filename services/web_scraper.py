import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import asyncio
import aiohttp
from urllib.parse import urlparse
import time
import random

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = None

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_page(self, url: str) -> Dict:
        """Busca o conteúdo de uma página de forma assíncrona"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove scripts e styles
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extrai o texto principal
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    return {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'content': text[:2000]
                    }
                return {'url': url, 'error': f'Status code: {response.status}'}
        except Exception as e:
            return {'url': url, 'error': str(e)}

    async def search_and_scrape(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Busca resultados usando DuckDuckGo e faz scraping do conteúdo
        """
        await self.init_session()
        
        try:
            # Busca resultados no DuckDuckGo
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    print("Search successful for query: ", query)
                    
                    # Extrai URLs dos resultados
                    results = []
                    for result in soup.select('.result__url'):
                        url = result.get('href')
                        if url and not url.startswith('/'):
                            results.append(url)
                            if len(results) >= max_results:
                                break
                    
                    # Faz scraping do conteúdo de cada URL
                    tasks = []
                    for url in results:
                        # Adiciona um pequeno delay aleatório para evitar bloqueios
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                        tasks.append(self.fetch_page(url))
                    
                    scraped_results = await asyncio.gather(*tasks)
                    return scraped_results
                
                return [{'error': f'Erro na busca: Status code {response.status}'}]
                
        except Exception as e:
            return [{'error': f'Erro geral: {str(e)}'}]
        finally:
            await self.close_session()

async def get_search_results(query: str, max_results: int = 20) -> List[Dict]:
    """
    Função principal para buscar e fazer scraping de resultados
    """
    scraper = WebScraper()
    return await scraper.search_and_scrape(query, max_results)

# Exemplo de uso
if __name__ == "__main__":
    async def main():
        results = await get_search_results("Brendi Brasil", max_results=5)
        for result in results:
            print(f"\nURL: {result.get('url', 'N/A')}")
            print(f"Título: {result.get('title', 'N/A')}")
            print(f"Conteúdo: {result.get('content', 'N/A')[:10000]}...")
            print("-" * 80)
        results = await get_search_results("Mercado Brasileiro de restaurantes vendas online", max_results=5)
        for result in results:
            print(f"\nURL: {result.get('url', 'N/A')}")
            print(f"Título: {result.get('title', 'N/A')}")
            print(f"Conteúdo: {result.get('content', 'N/A')[:10000]}...")
            print("-" * 80)

    asyncio.run(main()) 