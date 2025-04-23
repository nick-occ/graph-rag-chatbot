import os
import logging
from retry import retry
from neo4j import GraphDatabase
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import polars as pl
import pandas as pd

ARTICLES_CSV_PATH = os.getenv("ARTICLES_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

embeddings = OpenAIEmbeddings()

ARTICLE_NODE = "Article"
ARTICLE_CHUNK_NODE = "ArticleChunk"
HAS_CHUNK_RELATIONSHIP = "HAS_CHUNK"
ARTICLE_CHUNK_INDEX = "article_chunk_index"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


# function to get which articles are already chunked
def get_distinct_article_ids(tx):
    result = tx.run(
        f"""
        MATCH (c:{ARTICLE_CHUNK_NODE})
        RETURN DISTINCT c.article_id AS article_id
    """
    )
    return [record["article_id"] for record in result]


@retry(tries=100, delay=10)
def load_article_graph_from_csv() -> None:
    # load articles dataset
    articles = (
        pl.read_csv(ARTICLES_CSV_PATH, schema_overrides={"article_date": pl.Date})
        .filter(pl.col("article_text").is_not_null())
        .with_columns(pl.col("article_date").cast(pl.String))
    )

    LOGGER.info("Loaded articles dataset")

    # character splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""]
    )

    # create chunked text
    chunked = []
    for _, row in articles.to_pandas().iterrows():
        chunks = splitter.split_text(str(row["article_text"]))
        for i, chunk in enumerate(chunks):
            chunked.append(
                {
                    "id": f"{row['article_id']}_{i}",
                    "article_id": row["article_id"],
                    "text": chunk,
                    "chunk_index": i,
                }
            )

    LOGGER.info("Chunked articles")

    # create constraint for article node
    driver.execute_query(
        f"CREATE CONSTRAINT \
        IF NOT EXISTS FOR (a:{ARTICLE_NODE}) \
            REQUIRE a.id IS UNIQUE;"
    )

    LOGGER.info("Create unique constraint for Article node")

    # create dictionary of articles with fields needed for the node
    article_dicts = articles.select(
        pl.col("article_id").alias("id"),
        "title",
        pl.col("article_date").alias("date"),
        pl.col("article_url").alias("url"),
    ).to_dicts()

    # merge article nodes in database
    with driver.session() as session:
        for a in article_dicts:
            _cypher = f"""merge (a:{ARTICLE_NODE} {{id: $id}})
    set a.title = $title,
    a.date = $date,
    a.url = $url;
    """
            session.run(_cypher, **a)

        LOGGER.info("Created Article nodes")

    # create constraint for article chunk node
    driver.execute_query(
        f"CREATE CONSTRAINT \
        IF NOT EXISTS FOR (a:{ARTICLE_CHUNK_NODE}) \
            REQUIRE a.id IS UNIQUE;"
    )

    LOGGER.info("Create unique constraint for ArticleChunk node")

    # get articles that are already chunked
    with driver.session() as session:
        articles_chunked = session.execute_read(get_distinct_article_ids)

    article_chunks_to_process = [
        c for c in chunked if c["article_id"] not in articles_chunked
    ]
    LOGGER.info(f"There are {len(article_chunks_to_process):,} nodes to process")

    # create vector index with embedding
    if article_chunks_to_process:
        Neo4jVector.from_documents(
            documents=[
                Document(
                    page_content=c["text"],
                    metadata={
                        "id": c["id"],
                        "article_id": c["article_id"],
                        "chunk_index": c["chunk_index"],
                    },
                )
                for c in article_chunks_to_process
            ],
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=ARTICLE_CHUNK_INDEX,
            node_label=ARTICLE_CHUNK_NODE,
            text_node_property="text",
            embedding_node_property="embedding",
        )

        LOGGER.info("Vector index created for ArticleChunk node")

        with driver.session() as session:
            for chunk in article_chunks_to_process:
                session.run(
                    f"""
                match(a:{ARTICLE_NODE} {{id: $article_id}})
                match(c:{ARTICLE_CHUNK_NODE} {{id: $id}})
                merge(a)-[:{HAS_CHUNK_RELATIONSHIP}]->(c)
                """,
                    **chunk,
                )
            LOGGER.info("Created relationship between Article and Article Chunk")


if __name__ == "__main__":
    load_article_graph_from_csv()
