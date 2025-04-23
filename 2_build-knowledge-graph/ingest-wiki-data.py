import os
import time
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
    create_cypher_query_to_insert_wiki_article,
    create_cypher_query_for_wiki_links,
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
    index_name="category_embedding_index",  # Changed to match arXiv index name
    node_label="Category",
    text_node_properties=["name", "description"],
    embedding_node_property="embedding",
)

# Create necessary indices
for q in create_indices_queries():
    graph.query(q)

# Create additional indices for Wikipedia data (now using Paper node label)
graph.query("CREATE TEXT INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)")
graph.query("CREATE TEXT INDEX category_name IF NOT EXISTS FOR (c:Category) ON (c.code)")

# Main data ingestion process
print("Starting Wikipedia data ingestion for oil and gas articles...")

# Get seed articles
seed_articles = get_seed_oil_gas_articles()
print(f"Seed articles: {len(seed_articles)}")

# Set to track all articles we'll add
all_article_titles = set(seed_articles)
articles_to_process = []
page_id_to_arxiv_id = {}  # Map to track page_id to arxiv_id for citations

# First fetch all seed articles
for title in seed_articles:
    try:
        article = fetch_wikipedia_article(title)
        if article:
            articles_to_process.append(article)
            # Store mapping of title->arxiv_id for later citation creation
            page_id_to_arxiv_id[article.title] = article.arxiv_id
            print(f"Added seed article: {title}")
        else:
            print(f"Could not fetch seed article: {title}")
        # Add a delay to avoid API rate limits
        time.sleep(1)
    except Exception as e:
        print(f"Error fetching seed article {title}: {e}")

# Now fetch related articles (up to a limit)
related_articles_limit = 20  # Limit related articles per seed to avoid too many

for article in articles_to_process.copy():
    linked_articles = get_linked_articles(article.title)
    
    # Filter links to keep only those that might be relevant to oil & gas
    relevant_keywords = [" "]
    # ["oil", "gas", "petroleum", "energy", "drilling", "refinery", 
    #                      "fuel", "hydrocarbon", "pipeline", "opec", "reservoir", 
    #                      "offshore", "extraction", "well", "natural", "production", 
    #                      "exploration", "geology", "industry"]
    
    relevant_links = []
    for link in linked_articles:
        # Check if the link contains any relevant keywords
        if any(keyword.lower() in link.lower() for keyword in relevant_keywords):
            relevant_links.append(link)
    
    # Limit the number of related articles
    relevant_links = relevant_links[:related_articles_limit]
    
    # Initialize cited_arxiv_papers if not exists
    if not hasattr(article, 'cited_arxiv_papers'):
        article.cited_arxiv_papers = []
    
    # Add relevant links to titles to process
    for link in relevant_links:
        if link not in all_article_titles:
            all_article_titles.add(link)
            try:
                linked_article = fetch_wikipedia_article(link)
                if linked_article:
                    # Store mapping for later citation relationships
                    page_id_to_arxiv_id[linked_article.title] = linked_article.arxiv_id
                    
                    # Add to the parent article's cited papers list
                    article.cited_arxiv_papers.append(linked_article.arxiv_id)
                    
                    articles_to_process.append(linked_article)
                    print(f"Added related article: {link}")
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching related article {link}: {e}")
    
    # Update the article's linked_pages to use arxiv_ids instead of titles
    article.linked_pages = relevant_links  # Keep track of original links
    article.cited_arxiv_papers = [page_id_to_arxiv_id.get(title, "") for title in relevant_links if title in page_id_to_arxiv_id]

print(f"Total articles to insert: {len(articles_to_process)}")

# Insert all categories
all_categories = []
for article in articles_to_process:
    all_categories.extend(article.categories)
all_categories = list(set(all_categories))

category_query = """
UNWIND $categories AS category
MERGE (c:Category {code: category})
"""

try:
    graph.query(category_query, {"categories": all_categories})
    print(f"Inserted {len(all_categories)} categories")
except Exception as e:
    print(f"Error inserting categories: {e}")
    # Process categories one by one as a fallback
    successful_inserts = 0
    for category in all_categories:
        try:
            single_query = f"""
            MERGE (c:Category {{code: "{sanitize(category)}"}})
            """
            graph.query(single_query)
            successful_inserts += 1
        except Exception as inner_e:
            print(f"Error inserting category '{category}': {inner_e}")
    print(f"Successfully inserted {successful_inserts} out of {len(all_categories)} categories individually")
# Insert all articles as Paper nodes (not WikiPage)
for article in articles_to_process:
    try:
        query = create_cypher_query_to_insert_wiki_article(article)
        graph.query(query)
        print(f"Inserted article: {article.title} as Paper node with id: {article.arxiv_id}")
    except Exception as e:
        print(f"Error inserting article {article.title}: {e}")

# Process each article to create citation relationships (CITES instead of LINKS_TO)
for article in articles_to_process:
    try:
        # Get arxiv_ids for the related pages that we've actually added
        target_ids = []
        for cited_id in article.cited_arxiv_papers:
            if cited_id:  # Ensure non-empty ID
                target_ids.append(f'"{cited_id}"')
        
        if target_ids:
            query = create_cypher_query_for_wiki_links(article.arxiv_id, "[" + ",".join(target_ids) + "]")
            graph.query(query)
            print(f"Created citation relationships for article: {article.title}")
    except Exception as e:
        print(f"Error creating citation relationships for article {article.title}: {e}")

# Update citation counts after creating all relationships
for article in articles_to_process:
    citation_count_query = f"""
    MATCH (p:Paper)-[:CITES]->(target:Paper)
    WHERE target.id = '{article.arxiv_id}'
    RETURN COUNT(p) as citation_count
    """
    try:
        results = graph.query(citation_count_query)
        if results and len(results) > 0:
            citation_count = results[0]["citation_count"]
            # Update in-memory object
            article.citation_count = citation_count
            
            # Also update the count in the database
            update_query = f"""
            MATCH (p:Paper {{id: '{article.arxiv_id}'}})
            SET p.citation_count = {citation_count}
            """
            graph.query(update_query)
            print(f"Updated citation count for {article.title}: {citation_count}")
    except Exception as e:
        print(f"Error updating citation count for {article.title}: {e}")

# Process the content into chunks for vector search
raw_docs = []
for article in articles_to_process:
    # Extract clean text from HTML content
    clean_text = extract_clean_text_from_html(article.content)
    
    # Create document for ingestion
    raw_docs.append(
        Document(page_content=clean_text, metadata={"arxiv_id": article.arxiv_id})
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
            node_label="Chunk",  # Same as arXiv chunks
            index_name="chunk_embedding_index",  # Standardized index name
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
                    index_name="chunk_embedding_index",
                )
            except Exception as inner_e:
                print(f"Error in inserting chunk: {inner_e}")
    document_batch.clear()

# Link the chunks to the articles - using the same relationship as arXiv
graph.query(
    """
    MATCH (p:Paper), (c:Chunk)
    WHERE p.id = c.arxiv_id
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
    MATCH (p:Paper)
    WHERE p.page_id is not null
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
    MATCH (source:Paper)-[r:CITES]->(target:Paper)
    WHERE source.page_id IS NOT NULL AND target.page_id IS NOT NULL
    RETURN COUNT(r) as link_count
    """
)[0]["link_count"]

print("\nKnowledge Graph Summary:")
print(f"Total Wikipedia Articles: {article_count}")
print(f"Total Categories: {category_count}")
print(f"Total Links between Articles: {link_count}")
print(f"Total Text Chunks: {chunk_count}")
print("\nWikipedia Knowledge Graph ingestion complete!")