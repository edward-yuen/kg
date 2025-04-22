import json
import re
import requests
from datetime import date, datetime
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from langchain.graphs import Neo4jGraph

from utils.data_utils import sanitize


class IngestablePaper:
    def __init__(
        self,
        arxiv_id: str,
        arxiv_link: str,
        title: str,
        summary: str,
        authors: List[str],
        categories: List[str],
        pdf_link: str,
        published_date: date,
        full_text: str,
        cited_arxiv_papers: List[str],
    ):
        self.arxiv_id = arxiv_id
        self.arxiv_link = arxiv_link
        self.title = title
        self.summary = summary
        self.authors = authors
        self.categories = categories
        self.pdf_link = pdf_link
        self.published_date = published_date
        self.full_text = full_text
        self.cited_arxiv_papers = cited_arxiv_papers
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

    # Get pages/papers that link to/cite this article
    def get_citing_papers(self) -> List["IngestablePaper"]:
        query = r"""MATCH (p:Paper)-[:CITES]->(cited:Paper)
        WHERE cited.id = '$id'
        RETURN {
        id: p.id, title: p.title, summary: p.summary, published: p.published, arxiv_link: p.arxiv_link, pdf_link: p.pdf_link, cited_arxiv_papers: p.cited_arxiv_papers,
        authors: COLLECT { MATCH (p)-->(a:Author) RETURN a.name },
        categories: COLLECT { MATCH (p)-->(c:Category) RETURN c.code },
        citations: COUNT { (p)<-[:CITES]-(:Paper) }
        } AS result
        ORDER BY result.citations DESC
        """.replace(
            "$id", self.arxiv_id
        )
        results = self.graph_db_instance.query(query)
        papers = list()
        for r in results:
            obj = r["result"]
            paper = IngestablePaper(
                arxiv_id=obj["id"],
                title=obj["title"],
                summary=obj["summary"],
                published_date=obj["published"].to_native(),
                arxiv_link=obj["arxiv_link"],
                pdf_link=obj["pdf_link"],
                authors=obj["authors"],
                categories=obj["categories"],
                cited_arxiv_papers=obj["cited_arxiv_papers"],
                full_text="",
            )
            paper.citation_count = obj["citations"]
            papers.append(paper)
        return papers

    def get_top_authors(self) -> List[str]:
        query = r"""MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
        WHERE p.id='$id'
        WITH DISTINCT a
        RETURN {
            name: a.name,
            paper_ids: COLLECT {MATCH (op:Paper)-[:AUTHORED_BY]->(a:Author) RETURN op.id }
        } AS result
        ORDER BY SIZE(result.paper_ids) DESC
        """.replace(
            "$id", self.arxiv_id
        )
        results = self.graph_db_instance.query(query)
        return [r["result"]["name"] for r in results]


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
        contributors: List[str] = None,
    ):
        self.page_id = page_id
        self.title = title
        self.url = url
        self.summary = summary
        self.categories = categories
        self.content = content
        self.last_modified = last_modified
        self.linked_pages = linked_pages or []
        self.contributors = contributors or []
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

    # Convert WikipediaArticle to IngestablePaper for compatibility
    def to_ingestable_paper(self) -> IngestablePaper:
        """Convert a WikipediaArticle to an IngestablePaper for graph DB compatibility"""
        # Clean HTML content to plain text
        plain_text = extract_clean_text_from_html(self.content)
        
        return IngestablePaper(
            arxiv_id=self.page_id,  # Use page_id as the unique identifier
            arxiv_link=self.url,    # Original article URL
            title=self.title,
            summary=self.summary,
            authors=self.contributors,
            categories=self.categories,
            pdf_link=self.url,      # No direct PDF for Wikipedia, use the same URL
            published_date=self.last_modified,
            full_text=plain_text,
            cited_arxiv_papers=self.linked_pages,  # Linked pages as "citations"
        )


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
        time.sleep(5)  # Add delay to avoid rate limiting
        
        if summary_response.status_code != 200:
            print(f"Error fetching article {title}: {summary_response.status_code}")
            return None
        
        summary_data = summary_response.json()
        
        # Get full HTML content
        content_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded_title}"
        content_response = requests.get(content_url)
        time.sleep(5)  # Add delay to avoid rate limiting
        
        if content_response.status_code != 200:
            print(f"Error fetching content for {title}: {content_response.status_code}")
            return None
        
        content_html = content_response.text
        
        # Get contributors/authors
        contributors_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=contributors&format=json&titles={encoded_title}&pclimit=5"
        contributors_response = requests.get(contributors_url)
        time.sleep(5)  # Add delay to avoid rate limiting
        
        contributors = []
        if contributors_response.status_code == 200:
            contributors_data = contributors_response.json()
            page_id = list(contributors_data['query']['pages'].keys())[0]
            if 'contributors' in contributors_data['query']['pages'][page_id]:
                contributors = [contrib['name'] for contrib in contributors_data['query']['pages'][page_id]['contributors'][:5]]
        
        # If no contributors found, add a placeholder
        if not contributors:
            contributors = ["Wikipedia Contributors"]
        
        # Convert modified date from string to date object
        modified_date = datetime.strptime(summary_data.get('timestamp', '2000-01-01T00:00:00Z'), '%Y-%m-%dT%H:%M:%SZ').date()
        
        # We'll generate categories using the LLM later
        # Create the article object with empty categories for now
        return WikipediaArticle(
            page_id=str(summary_data.get('pageid', 0)),
            title=summary_data.get('title', ''),
            url=summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
            summary=summary_data.get('extract', ''),
            categories=[],  # Empty categories to be filled by LLM
            content=content_html,
            last_modified=modified_date,
            linked_pages=[],
            contributors=contributors,
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

def get_linked_page_ids(articles_to_process: List[WikipediaArticle]) -> Dict[str, str]:
    """Create a mapping from article title to page_id for linked pages"""
    # First build a map of all known titles to their page IDs
    title_to_id_map = {}
    for article in articles_to_process:
        title_to_id_map[article.title] = article.page_id
    
    return title_to_id_map

def create_wiki_paper_object(title: str) -> Optional[IngestablePaper]:
    """Create an IngestablePaper object from a Wikipedia article title"""
    # Fetch the article data
    wiki_article = fetch_wikipedia_article(title)
    if not wiki_article:
        return None
    
    # Get linked articles
    linked_articles = get_linked_articles(title)
    wiki_article.linked_pages = linked_articles
    
    # Convert to IngestablePaper format for compatibility with existing pipeline
    return wiki_article.to_ingestable_paper()


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