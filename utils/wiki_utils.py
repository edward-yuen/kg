import json
import re
import time
import requests
from datetime import date, datetime
from typing import Dict, List, Optional
from functools import lru_cache

from bs4 import BeautifulSoup
from langchain.graphs import Neo4jGraph
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate

from utils.data_utils import sanitize
import utils.constants as const
from utils.huggingface_utils import load_local_model


# Lazy-loaded LLM instance
_llm_instance = None

def get_llm() -> BaseLLM:
    """Lazy-load the LLM only when needed"""
    global _llm_instance
    if _llm_instance is None:
        print("Initializing LLM for category generation...")
        _llm_instance = load_local_model()
    return _llm_instance


def generate_categories_with_llm(article_title: str, article_summary: str) -> List[str]:
    """Generate categories for a Wikipedia article using an LLM"""
    
    # Get the LLM instance (will be initialized if needed)
    llm = get_llm()
    
    prompt_template = """<|start_header_id|>system<|end_header_id|>
You are an AI language model that specializes in categorizing content. Your task is to generate appropriate categories for Wikipedia articles in a format similar to arXiv categories.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Please generate 3-5 categories for the following Wikipedia article. 
The categories should follow this format: domain.subdomain (e.g., cs.AI, math.OC, physics.fluid-dyn)
Where:
- domain is a broader field (cs, math, physics, bio, econ, etc.)
- subdomain is a more specific area within that field

Format your response as a comma-separated list without any additional text or explanation.

Article Title: {title}
Article Summary: {summary}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    prompt = PromptTemplate.from_template(const.llama3_bos_token + prompt_template)
    chain = prompt | llm
    
    response = chain.invoke({
        "title": article_title,
        "summary": article_summary,
    })
    
    # Clean up the response and split by commas
    categories = [cat.strip() for cat in response.split(',')]
    
    # Filter out empty strings and ensure valid format
    valid_categories = []
    for cat in categories:
        if cat and '.' in cat:  # Basic check that it has a domain.subdomain format
            # Remove any characters that might cause Cypher issues
            valid_categories.append(cat)
    
    return valid_categories


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
        # Original WikiPage properties
        self.page_id = page_id
        self.title = title
        self.url = url
        self.summary = summary
        self.categories = categories
        self.content = content
        self.last_modified = last_modified
        self.linked_pages = linked_pages or []
        
        # Generate arxiv_id format: YYMM.NNNNN
        # Use year and month from last_modified
        year_month = f"{last_modified.year % 100:02d}{last_modified.month:02d}"  # e.g., 2404 for 2024 April
        
        # Convert page_id to a number and take the last 5 digits, with zero padding
        try:
            page_num = int(page_id) % 100000  # Take last 5 digits
        except ValueError:
            # If page_id is not a number, hash it and take the last 5 digits
            import hashlib
            hash_obj = hashlib.md5(page_id.encode())
            page_num = int(hash_obj.hexdigest(), 16) % 100000
            
        self.arxiv_id = f"{year_month}.{page_num:05d}"  # Ensure 5 digits with zero padding
        
        # Add arXiv-compatible properties
        self.published_date = last_modified  # Map last_modified to published_date
        self.arxiv_link = url  # Map url to arxiv_link
        self.pdf_link = url  # Map url to pdf_link
        self.authors = ["Wikipedia"]  # Default author
        self.full_text = content  # Map content to full_text
        self.cited_arxiv_papers = linked_pages  # Will be populated with synthetic arxiv_ids later
        
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

    # Get pages that link to this article - updated to use Paper nodes and CITES relationship
    def get_citing_papers(self) -> List["WikipediaArticle"]:
        query = r"""MATCH (p:Paper)-[:CITES]->(cited:Paper)
        WHERE cited.id = '$id'
        RETURN {
        id: p.id, title: p.title, summary: p.summary, published: p.published, 
        arxiv_link: p.arxiv_link, pdf_link: p.pdf_link, cited_arxiv_papers: p.cited_arxiv_papers,
        page_id: p.page_id, url: p.url,
        authors: COLLECT { MATCH (p)-[:AUTHORED_BY]->(a:Author) RETURN a.name },
        categories: COLLECT { MATCH (p)-[:HAS_CATEGORY]->(c:Category) RETURN c.code },
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
            paper = WikipediaArticle(
                page_id=obj.get("page_id", ""),
                title=obj["title"],
                summary=obj["summary"],
                last_modified=obj["published"].to_native(),
                url=obj.get("url", obj["arxiv_link"]),
                categories=obj["categories"],
                content="",
                linked_pages=obj.get("cited_arxiv_papers", []),
            )
            paper.citation_count = obj["citations"]
            papers.append(paper)
        return papers

    # Get categories this article belongs to - updated to use HAS_CATEGORY relationship
    def get_top_categories(self) -> List[str]:
        query = r"""MATCH (p:Paper)-[:HAS_CATEGORY]->(c:Category)
        WHERE p.id='$id'
        WITH DISTINCT c
        RETURN c.code AS category_code
        """.replace(
            "$id", self.arxiv_id
        )
        results = self.graph_db_instance.query(query)
        return [r["category_code"] for r in results]
        
    # Add a method to match IngestablePaper's get_top_authors
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


class WikipediaChunk:
    def __init__(self, text: str, article: WikipediaArticle):
        self.text = text
        self.paper = article  # Changed from article to paper to match PaperChunk
        self._metadata = dict()

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> Dict:
        self._metadata = value


def fetch_wikipedia_article(title: str) -> Optional[WikipediaArticle]:
    """Fetch article data from Wikipedia API and generate categories using LLM"""
    # Encode the title for URLs
    encoded_title = requests.utils.quote(title)
    
    try:
        # Get the summary/metadata
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
        summary_response = requests.get(api_url)
        time.sleep(1)  # Add delay to avoid rate limiting
        
        if summary_response.status_code != 200:
            print(f"Error fetching article {title}: {summary_response.status_code}")
            return None
        
        summary_data = summary_response.json()
        
        # Get full HTML content
        content_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded_title}"
        content_response = requests.get(content_url)
        time.sleep(1)  # Add delay to avoid rate limiting
        
        if content_response.status_code != 200:
            print(f"Error fetching content for {title}: {content_response.status_code}")
            return None
        
        content_html = content_response.text
        
        # Generate categories using LLM
        categories = []
        if summary_data.get('extract'):
            try:
                categories = generate_categories_with_llm(
                    article_title=summary_data.get('title', ''),
                    article_summary=summary_data.get('extract', '')
                )
                print(f"Generated {len(categories)} categories for {title} using LLM")
            except Exception as e:
                print(f"Error generating categories with LLM for {title}: {e}")
        
        # Convert modified date from string to date object
        modified_date = datetime.strptime(summary_data.get('timestamp', '2000-01-01T00:00:00Z'), '%Y-%m-%dT%H:%M:%SZ').date()
        
        # Create the article object with the LLM-generated categories
        article = WikipediaArticle(
            page_id=str(summary_data.get('pageid', 0)),
            title=summary_data.get('title', ''),
            url=summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
            summary=summary_data.get('extract', ''),
            categories=categories,  # Using LLM-generated categories
            content=content_html,
            last_modified=modified_date,
            linked_pages=[],
        )
        
        return article
    
    except Exception as e:
        print(f"Error in fetching Wikipedia article for {title}: {e}")
        return None


def get_linked_articles(article_title: str) -> List[str]:
    """Get titles of articles linked from the given article"""
    encoded_title = requests.utils.quote(article_title)
    links_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=links&format=json&titles={encoded_title}&pllimit=500"
    
    try:
        response = requests.get(links_url)
        time.sleep(1)  # Add delay to avoid rate limiting
        
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


def create_cypher_query_to_insert_wiki_article(article: WikipediaArticle):
    """Create Cypher query to insert a Wikipedia article as a Paper node"""
    neo4j_date_string = f'date("{article.last_modified.strftime("%Y-%m-%d")}")'
    categories_neo4j = '["' + ('","').join([sanitize(c) for c in article.categories]) + '"]'
    linked_pages_neo4j = '["' + ('","').join([sanitize(p) for p in article.linked_pages]) + '"]'
    
    # Sanitize text fields for Neo4j
    title = sanitize(article.title)
    summary = sanitize(article.summary)
    content = sanitize(article.content[:10000]) if len(article.content) > 10000 else sanitize(article.content)
    
    query = f"""
    MERGE (page:Paper {{id: "{article.arxiv_id}"}})
    ON CREATE
    SET
        page.title = "{title}",
        page.summary = "{summary}",
        page.content = "{content}",
        page.published = {neo4j_date_string},
        page.url = "{article.url}",
        page.arxiv_link = "{article.url}",
        page.pdf_link = "{article.url}",
        page.cited_arxiv_papers = {linked_pages_neo4j},
        page.page_id = "{article.page_id}"
    
    FOREACH (category in {categories_neo4j} | 
        MERGE (c:Category {{code: category}})
        MERGE (page)-[:HAS_CATEGORY]->(c)
    )
    
    WITH page
    
    MERGE (author:Author {{name: "Wikipedia"}})
    MERGE (page)-[:AUTHORED_BY]->(author)
    """
    return query


def create_cypher_query_for_wiki_links(source_id, target_ids):
    """Create Cypher query for citation relationships between articles"""
    query = f"""
    MATCH (source:Paper {{id: "{source_id}"}})
    MATCH (target:Paper)
    WHERE target.id IN {str(target_ids)}
    MERGE (source)-[:CITES]->(target)
    """
    return query