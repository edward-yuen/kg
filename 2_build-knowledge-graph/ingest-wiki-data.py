import os
import time
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector

import utils.constants as const
from utils.data_utils import (
    create_cypher_batch_query_to_create_citation_relationship,
    create_cypher_batch_query_to_insert_arxiv_papers,
    create_indices_queries,
    sanitize
)
from utils.huggingface_utils import cache_and_load_embedding_model, load_local_model
from utils.neo4j_utils import (
    get_neo4j_credentails,
    is_neo4j_server_up,
    reset_neo4j_server,
    wait_for_neo4j_server,
)
from utils.wiki_utils import (
    IngestablePaper,
    WikipediaArticle,
    create_wiki_paper_object,
    get_linked_articles,
    get_linked_page_ids,
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

# Load local LLM for category generation
print("Loading local LLM for category generation...")
llm = load_local_model()
print("Local LLM loaded successfully.")

# Create category insertion query for oil & gas categories
# Replace the existing create_query_for_wiki_category_insertion function with this one:

def create_query_for_wiki_category_insertion(categories):
    """Create Cypher query to insert Wikipedia categories with better error handling"""
    queries = []
    for i, category in enumerate(categories):
        # Skip empty categories
        if not category or len(category.strip()) == 0:
            continue
            
        # Ensure category is properly sanitized
        safe_category = sanitize(category)
        
        # Skip if category became empty after sanitization
        if not safe_category:
            continue
            
        # Create a separate query for each category for better error handling
        queries.append(f"""
            MERGE (category_{i}:Category {{code: "{safe_category}"}})
            ON CREATE SET
                category_{i}.title = "{safe_category}",
                category_{i}.description = "Oil and Gas category"
        """)
    
    return queries

# Then modify the category insertion part of the script:

# Get all unique categories
all_categories = []
for paper in papers_to_insert:
    all_categories.extend(paper.categories)
all_categories = list(set(all_categories))
print(f"Found {len(all_categories)} unique categories")

# Insert all categories
category_queries = create_query_for_wiki_category_insertion(all_categories)
successful_categories = 0
for query in category_queries:
    try:
        graph.query(query)
        successful_categories += 1
    except Exception as e:
        print(f"Error inserting category: {e}")
        continue
print(f"Inserted {successful_categories} out of {len(all_categories)} categories")

# Main data ingestion process
print("Starting Wikipedia data ingestion for oil and gas articles...")

# Get seed articles
seed_articles = get_seed_oil_gas_articles()
print(f"Seed articles: {len(seed_articles)}")

# Set to track all articles we'll add
all_article_titles = set(seed_articles)
papers_to_insert = []

# First fetch all seed articles and convert to IngestablePaper format
for title in seed_articles:
    try:
        # Use the LLM to generate categories
        paper = create_wiki_paper_object(title, llm)
        if paper:
            papers_to_insert.append(paper)
            print(f"Added seed article: {title} with categories: {paper.categories}")
        else:
            print(f"Could not fetch seed article: {title}")
        # Add a delay to avoid API rate limits
        time.sleep(2)
    except Exception as e:
        print(f"Error fetching seed article {title}: {e}")

# Now fetch related articles (up to a limit)
related_articles_limit = 10  # Limit related articles per seed to avoid too many

for paper in papers_to_insert.copy():
    # The paper's cited_arxiv_papers field contains the linked page titles
    linked_articles = paper.cited_arxiv_papers
    
    # Filter links to keep only those that might be relevant to oil & gas
    relevant_keywords = [
        "oil", "gas", "petroleum", "energy", "drilling", "refinery", 
        "fuel", "hydrocarbon", "pipeline", "opec", "reservoir", 
        "offshore", "extraction", "well", "natural", "production", 
        "exploration", "geology", "industry"
    ]
    
    relevant_links = []
    for link in linked_articles:
        # Check if the link contains any relevant keywords
        if any(keyword.lower() in link.lower() for keyword in relevant_keywords):
            relevant_links.append(link)
    
    # Limit the number of related articles
    relevant_links = relevant_links[:related_articles_limit]
    
    # Update the paper's cited papers list to contain only relevant links
    paper.cited_arxiv_papers = relevant_links
    
    # Add relevant links to titles to process
    for link in relevant_links:
        if link not in all_article_titles:
            all_article_titles.add(link)
            try:
                # Use the LLM to generate categories for linked articles
                linked_paper = create_wiki_paper_object(link, llm)
                if linked_paper:
                    papers_to_insert.append(linked_paper)
                    print(f"Added related article: {link} with categories: {linked_paper.categories}")
                # Add a delay to avoid API rate limits
                time.sleep(2)
            except Exception as e:
                print(f"Error fetching related article {link}: {e}")

print(f"Total articles to insert: {len(papers_to_insert)}")

# Create indices
for q in create_indices_queries():
    graph.query(q)
    
# Get all unique categories
all_categories = []
for paper in papers_to_insert:
    all_categories.extend(paper.categories)
all_categories = list(set(all_categories))

# Insert all categories
category_query = create_query_for_wiki_category_insertion(all_categories)
graph.query(category_query)
print(f"Inserted {len(all_categories)} categories")

# Insert all articles in batches
paper_batch = []
batch_size = 10
for i, paper in enumerate(papers_to_insert):
    if len(paper_batch) < batch_size and i != len(papers_to_insert) - 1:
        paper_batch.append(paper)
        continue
    
    paper_batch.append(paper)
    query = create_cypher_batch_query_to_insert_arxiv_papers(paper_batch)
    
    try:
        graph.query(query)
        print(f"Inserted papers {[p.arxiv_id for p in paper_batch]}")
    except Exception as e:
        print(f"Error in batch insert: {e}")
        # Try one by one if batch fails
        for p in paper_batch:
            query = create_cypher_batch_query_to_insert_arxiv_papers([p])
            try:
                graph.query(query)
                print(f"Inserted paper {p.arxiv_id}")
            except Exception as e:
                print(f"Error inserting paper {p.arxiv_id}: {e}")
    
    paper_batch.clear()

# Create citation relationships
for paper in papers_to_insert:
    query = create_cypher_batch_query_to_create_citation_relationship(paper.arxiv_id)
    try:
        graph.query(query)
        print(f"Created citation relationships for paper {paper.arxiv_id}")
    except Exception as e:
        print(f"Error creating citation relationships for {paper.arxiv_id}: {e}")

# Process the content into chunks for vector search
raw_docs = [
    Document(page_content=p.full_text, metadata={"arxiv_id": p.arxiv_id})
    for p in papers_to_insert
]

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
                )
            except Exception as inner_e:
                print(f"Error inserting chunk: {inner_e}")
    document_batch.clear()

# Link the chunks to the papers
graph.query(
    """
MATCH (p:Paper), (c:Chunk)
WHERE p.id = c.arxiv_id
MERGE (p)-[:CONTAINS_TEXT]->(c)
"""
)

# Delete orphan chunks with no papers
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
MATCH (p:Paper)
RETURN COUNT(p) as article_count
"""
)[0]["article_count"]

category_count = graph.query(
    """
MATCH (c:Category)
RETURN COUNT(c) as category_count
"""
)[0]["category_count"]

citation_count = graph.query(
    """
MATCH ()-[r:CITES]->()
RETURN COUNT(r) as citation_count
"""
)[0]["citation_count"]

author_count = graph.query(
    """
MATCH (a:Author)
RETURN COUNT(a) as author_count
"""
)[0]["author_count"]

print("\nKnowledge Graph Summary:")
print(f"Total Wikipedia Articles: {article_count}")
print(f"Total Categories: {category_count}")
print(f"Total Citations between Articles: {citation_count}")
print(f"Total Authors: {author_count}")
print(f"Total Text Chunks: {chunk_count}")
print("\nWikipedia Knowledge Graph ingestion complete!")