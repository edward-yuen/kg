import json
import re
import requests
from datetime import date, datetime
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from langchain.graphs import Neo4jGraph

from utils.data_utils import sanitize


class WikipediaArticle:
    def __init__(
        self,
        page_id: str,
        title: str,
        url: str,
        summary: str,
        categories: List[str],
        content: str,
        last_modified: date,
        linked_pages: List[str] = None,
    ):
        self.page_id = page_id
        self.title = title
        self.url = url
        self.summary = summary
        self.categories = categories
        self.content = content
        self.last_modified = last_modified
        self.linked_pages = linked_pages or []
        self._citation_count = None
        self._graph_db_instance = None

    @property
    def citation_count(self):
        return self._citation_count

    @citation_count.setter
    def citation_count(self, value):
        self._citation_count = value

    @property
    def graph_db_instance(self) -> Neo4jGraph:
        return self._graph_db_instance

    @graph_db_instance.setter
    def graph_db_instance(self, value):
        self._graph_db_instance = value

    # Get pages that link to this article
    def get_linking_pages(self) -> List["WikipediaArticle"]:
        query = r"""MATCH (p:WikiPage)-[:LINKS_TO]->(cited:WikiPage)
        WHERE cited.page_id = '$id'
        RETURN {
        page_id: p.page_id, title: p.title, summary: p.summary, last_modified: p.last_modified, url: p.url, content: p.content,
        categories: COLLECT { MATCH (p)-->(c:Category) RETURN c.name },
        links_count: COUNT { (p)<-[:LINKS_TO]-(:WikiPage) }
        } AS result
        ORDER BY result.links_count DESC
        """.replace(
            "$id", self.page_id
        )
        results = self.graph_db_instance.query(query)
        pages = []
        for r in results:
            obj = r["result"]
            page = WikipediaArticle(
                page_id=obj["page_id"],
                title=obj["title"],
                summary=obj["summary"],
                last_modified=obj["last_modified"].to_native(),
                url=obj["url"],
                categories=obj["categories"],
                content=obj["content"],
                linked_pages=[],
            )
            page.citation_count = obj["links_count"]
            pages.append(page)
        return pages

    # Get categories this article belongs to
    def get_top_categories(self) -> List[str]:
        query = r"""MATCH (p:WikiPage)-[:BELONGS_TO]->(c:Category)
        WHERE p.page_id='$id'
        WITH DISTINCT c
        RETURN c.name AS category_name
        """.replace(
            "$id", self.page_id
        )
        results = self.graph_db_instance.query(query)
        return [r["category_name"] for r in results]


class WikipediaChunk:
    def __init__(self, text: str, article: WikipediaArticle):
        self.text = text
        self.article = article
        self._metadata = dict()

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> Dict:
        self._metadata = value


def fetch_wikipedia_article(title: str) -> Optional[WikipediaArticle]:
    """Fetch article data from Wikipedia API"""
    # Encode the title for URLs
    encoded_title = requests.utils.quote(title)
    
    try:
        # Get the summary/metadata
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
        summary_response = requests.get(api_url)
        if summary_response.status_code != 200:
            print(f"Error fetching article {title}: {summary_response.status_code}")
            return None
        
        summary_data = summary_response.json()
        
        # Get full HTML content
        content_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded_title}"
        content_response = requests.get(content_url)
        
        if content_response.status_code != 200:
            print(f"Error fetching content for {title}: {content_response.status_code}")
            return None
        
        content_html = content_response.text
        
        # Get categories
        categories_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=categories&format=json&titles={encoded_title}"
        categories_response = requests.get(categories_url)
        
        categories = []
        if categories_response.status_code == 200:
            categories_data = categories_response.json()
            page_id = list(categories_data['query']['pages'].keys())[0]
            if 'categories' in categories_data['query']['pages'][page_id]:
                categories = [cat['title'].replace('Category:', '') for cat in categories_data['query']['pages'][page_id]['categories']]
        
        # Convert modified date from string to date object
        modified_date = datetime.strptime(summary_data.get('timestamp', '2000-01-01T00:00:00Z'), '%Y-%m-%dT%H:%M:%SZ').date()
        
        # Create the article object
        return WikipediaArticle(
            page_id=str(summary_data.get('pageid', 0)),
            title=summary_data.get('title', ''),
            url=summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
            summary=summary_data.get('extract', ''),
            categories=categories,
            content=content_html,
            last_modified=modified_date,
            linked_pages=[],
        )
    
    except Exception as e:
        print(f"Error in fetching Wikipedia article for {title}: {e}")
        return None


def get_linked_articles(article_title: str) -> List[str]:
    """Get titles of articles linked from the given article"""
    encoded_title = requests.utils.quote(article_title)
    links_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=links&format=json&titles={encoded_title}&pllimit=500"
    
    try:
        response = requests.get(links_url)
        if response.status_code != 200:
            return []
        
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        
        # Extract first page (there should only be one)
        if not pages:
            return []
        
        page_id = list(pages.keys())[0]
        links = pages[page_id].get('links', [])
        
        # Extract titles from links
        return [link.get('title') for link in links if 'title' in link]
    
    except Exception as e:
        print(f"Error getting links for {article_title}: {e}")
        return []


def extract_clean_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    # Get text
    text = soup.get_text(separator=' ')
    # Remove extra whitespace
    clean_text = ' '.join(text.split())
    return clean_text


def get_seed_oil_gas_articles() -> List[str]:
    """Return a list of seed articles related to oil and gas"""
    return [
        "Petroleum",
        "Natural gas",
        "Petroleum industry",
        "Oil well",
        "Drilling rig",
        "Oil refinery",
        "OPEC",
        "Oil platform",
        "Hydraulic fracturing",
        "Liquefied natural gas",
        "Oil shale",
        "Petrochemical",
        "Offshore drilling",
        "Oil pipeline",
        "Crude oil",
        "Enhanced oil recovery",
        "Oil sands",
        "Reservoir engineering",
        "Seismic survey",
        "Natural gas processing"
    ]


def linkify_wiki_titles(text: str) -> str:
    """Convert Wikipedia article titles in text to markdown links"""
    article_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    articles = re.findall(article_pattern, text)
    articles = list(set(articles))
    
    new_text = text
    for article in articles:
        encoded_article = requests.utils.quote(article)
        new_text = re.sub(
            r'\b' + re.escape(article) + r'\b',
            f"[{article}](https://en.wikipedia.org/wiki/{encoded_article})",
            new_text
        )
    
    return new_text