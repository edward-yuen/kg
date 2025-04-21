import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector

import utils.constants as const
from utils.data_utils import create_indices_queries, sanitize
from utils.huggingface_utils import cache_and_load_embedding_model
from utils.neo4j_utils import (
    get_neo4j_credentails,
    is_neo4j_server_up,
    reset_neo4j_server,
    wait_for_neo4j_server,
)
from utils.wiki_utils import (
    WikipediaArticle,
    fetch_wikipedia_article,
    get_linked_articles,
    get_seed_oil_gas_articles,
    extract_clean_text_from_html,
)

load_dotenv()

# Ensure Neo4j server is running
if not is_neo4j_server_up():
    reset_neo4j_server()
    wait_for_neo4j_server()

# Connect to Neo4j
graph = Neo4jGraph(
    username=get_neo4j_credentails()["username"],
    password=get_neo4j_credentails()["password"],
    url=get_neo4j_credentails()["uri"],
)

# Clear the database (be careful if you have other data)
graph.query("MATCH (n) DETACH DELETE n")

# Load embedding model
embedding = cache_and_load_embedding_model()

# Create vector index for categories
Neo4jVector.from_existing_graph(
    embedding=embedding,
    url=get_neo4j_credentails()["uri"],
    username=get_neo4j_credentails()["username"],
    password=get_neo4j_credentails()["password"],
    index_name="wiki_category_embedding_index",
    node_label="Category",
    text_node_properties=["name", "description"],
    embedding_node_property="embedding",
)

# Create necessary indices
for q in create_indices_queries():
    graph.query(q)

# Create additional indices for Wikipedia data
graph.query("CREATE TEXT INDEX wiki_page_id IF NOT EXISTS FOR (p:WikiPage) ON (p.page_id)")
graph.query("CREATE TEXT INDEX category_name IF NOT EXISTS FOR (c:Category) ON (c.name)")

# Functions to create Cypher queries

def create_query_for_wiki_category_insertion(categories):
    """Create Cypher query to insert Wikipedia categories"""
    query = ""
    for i, category in enumerate(categories):
        safe_category = sanitize(category)
        query += f"""
            MERGE (category_{i}:Category {{name: "{safe_category}"}})
        """
    return query

def create_cypher_query_to_insert_wiki_article(article: WikipediaArticle):
    """Create Cypher query to insert a Wikipedia article"""
    neo4j_date_string = f'date("{article.last_modified.strftime("%Y-%m-%d")}")'
    categories_neo4j = '["' + ('","').join([sanitize(c) for c in article.categories]) + '"]'
    linked_pages_neo4j = '["' + ('","').join([sanitize(p) for p in article.linked_pages]) + '"]'
    
    # Sanitize text fields for Neo4j
    title = sanitize(article.title)
    summary = sanitize(article.summary)
    # We'll store a truncated version of the content to save space
    content = sanitize(article.content[:10000]) if len(article.content) > 10000 else sanitize(article.content)
    
    query = f"""
    MERGE (page:WikiPage {{page_id: "{article.page_id}"}})
    ON CREATE
      SET
        page.title = "{title}",
        page.summary = "{summary}",
        page.content = "{content}",
        page.last_modified = {neo4j_date_string},
        page.url = "{article.url}",
        page.linked_pages = {linked_pages_neo4j}
    FOREACH (category in {categories_neo4j} | 
        MERGE (c:Category {{name: category}}) 
        MERGE (page)-[:BELONGS_TO]->(c)
    )
    """
    return query

def create_cypher_query_for_wiki_links(source_id, target_ids):
    """Create Cypher query for link relationships between articles"""
    query = f"""
    MATCH (source:WikiPage {{page_id: "{source_id}"}})
    MATCH (target:WikiPage)
    WHERE target.page_id IN {str(target_ids)}
    MERGE (source)-[:LINKS_TO]->(target)
    """
    return query

# Main data ingestion process
print("Starting Wikipedia data ingestion for oil and gas articles...")

# Get seed articles
seed_articles = get_seed_oil_gas_articles()
print(f"Seed articles: {len(seed_articles)}")

# Set to track all articles we'll add
all_article_titles = set(seed_articles)
articles_to_process = []

# First fetch all seed articles
for title in seed_articles:
    try:
        article = fetch_wikipedia_article(title)
        if article:
            articles_to_process.append(article)
            print(f"Added seed article: {title}")
        else:
            print(f"Could not fetch seed article: {title}")
    except Exception as e:
        print(f"Error fetching seed article {title}: {e}")

# Now fetch related articles (up to a limit)
related_articles_limit = 10  # Limit related articles per seed to avoid too many

for article in articles_to_process.copy():
    linked_articles = get_linked_articles(article.title)
    
    # Filter links to keep only those that might be relevant to oil & gas
    relevant_keywords = ["oil", "gas", "petroleum", "energy", "drilling", "refinery", 
                         "fuel", "hydrocarbon", "pipeline", "opec", "reservoir", 
                         "offshore", "extraction", "well", "natural", "production", 
                         "exploration", "geology", "industry"]
    
    relevant_links = []
    for link in linked_articles:
        # Check if the link contains any relevant keywords
        if any(keyword.lower() in link.lower() for keyword in relevant_keywords):
            relevant_links.append(link)
    
    # Limit the number of related articles
    relevant_links = relevant_links[:related_articles_limit]
    article.linked_pages = relevant_links
    
    # Add relevant links to titles to process
    for link in relevant_links:
        if link not in all_article_titles:
            all_article_titles.add(link)
            try:
                linked_article = fetch_wikipedia_article(link)
                if linked_article:
                    articles_to_process.append(linked_article)
                    print(f"Added related article: {link}")
            except Exception as e:
                print(f"Error fetching related article {link}: {e}")

print(f"Total articles to insert: {len(articles_to_process)}")

# Insert all categories
all_categories = []
for article in articles_to_process:
    all_categories.extend(article.categories)
all_categories = list(set(all_categories))
graph.query(create_query_for_wiki_category_insertion(all_categories))
print(f"Inserted {len(all_categories)} categories")

# Insert all articles
for article in articles_to_process:
    try:
        query = create_cypher_query_to_insert_wiki_article(article)
        graph.query(query)
        print(f"Inserted article: {article.title}")
    except Exception as e:
        print(f"Error inserting article {article.title}: {e}")

# Create page ID map for link relationships
page_id_map = {}  # Map from title to page_id
for article in articles_to_process:
    page_id_map[article.title] = article.page_id

# Process each article to create link relationships
for article in articles_to_process:
    try:
        # Get page_ids for the related pages that we've actually added
        target_ids = []
        for linked_title in article.linked_pages:
            if linked_title in page_id_map:
                target_ids.append(f'"{page_id_map[linked_title]}"')
        
        if target_ids:
            query = create_cypher_query_for_wiki_links(article.page_id, "[" + ",".join(target_ids) + "]")
            graph.query(query)
            print(f"Created link relationships for article: {article.title}")
    except Exception as e:
        print(f"Error creating link relationships for article {article.title}: {e}")

# Process the content into chunks for vector search
raw_docs = []
for article in articles_to_process:
    # Extract clean text from HTML content
    clean_text = extract_clean_text_from_html(article.content)
    
    # Create document for ingestion
    raw_docs.append(
        Document(page_content=clean_text, metadata={"page_id": article.page_id})
    )

# Define chunking strategy
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=20,
    disallowed_special=(),
)

# Chunk the documents
documents = text_splitter.split_documents(raw_docs)
print(f"Number of chunks to be inserted into the knowledge graph: {len(documents)}")

# Insert chunks in batches
document_batch = []
batch_size = 50
for i, doc in enumerate(documents):
    document_batch.append(doc)
    if len(document_batch) < batch_size and i != len(documents) - 1:
        continue
    try:
        Neo4jVector.from_documents(
            documents=document_batch,
            embedding=embedding,
            url=get_neo4j_credentails()["uri"],
            username=get_neo4j_credentails()["username"],
            password=get_neo4j_credentails()["password"],
            node_label="Chunk",
            index_name="wiki_chunk_embedding_index",
        )
        print(f"Inserted chunk {i+1}/{len(documents)}")
    except Exception as e:
        print(f"Error in batch insert: {e}")
        for d in document_batch:
            try:
                Neo4jVector.from_documents(
                    documents=[d],
                    embedding=embedding,
                    url=get_neo4j_credentails()["uri"],
                    username=get_neo4j_credentails()["username"],
                    password=get_neo4j_credentails()["password"],
                    node_label="Chunk",
                    index_name="wiki_chunk_embedding_index",
                )
            except Exception as inner_e:
                print(f"Error in inserting chunk: {inner_e}")
    document_batch.clear()

# Link the chunks to the articles
graph.query(
    """
    MATCH (p:WikiPage), (c:Chunk)
    WHERE p.page_id = c.page_id
    MERGE (p)-[:CONTAINS_TEXT]->(c)
    """
)

# Delete orphan chunks with no articles
graph.query(
    """
    MATCH (c:Chunk)
    WHERE NOT (c)<-[:CONTAINS_TEXT]-()
    DETACH DELETE c
    """
)

# Get the number of chunks finally present in the DB
chunk_count = graph.query(
    """
    MATCH (c:Chunk)
    RETURN COUNT(c) as chunk_count
    """
)[0]["chunk_count"]

print(f"Number of chunks inserted into the knowledge graph: {chunk_count}")

# Print summary statistics
article_count = graph.query(
    """
    MATCH (p:WikiPage)
    RETURN COUNT(p) as article_count
    """
)[0]["article_count"]

category_count = graph.query(
    """
    MATCH (c:Category)
    RETURN COUNT(c) as category_count
    """
)[0]["category_count"]

link_count = graph.query(
    """
    MATCH ()-[r:LINKS_TO]->()
    RETURN COUNT(r) as link_count
    """
)[0]["link_count"]

print("\nKnowledge Graph Summary:")
print(f"Total Wikipedia Articles: {article_count}")
print(f"Total Categories: {category_count}")
print(f"Total Links between Articles: {link_count}")
print(f"Total Text Chunks: {chunk_count}")
print("\nWikipedia Knowledge Graph ingestion complete!")