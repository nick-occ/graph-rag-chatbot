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
YEAR_NODE = "Year"
HAS_CHUNK_RELATIONSHIP = "HAS_CHUNK"
PUBLISHED_RELATIONSHIP = "PUBLISHED_ON"
ARTICLE_CHUNK_INDEX = "article_chunk_index"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


# function to get which articles are already chunked
def get_distinct_article_ids(tx):
    try:
        result = tx.run(
            f"""
            MATCH (c:{ARTICLE_CHUNK_NODE})
            RETURN DISTINCT c.article_id AS article_id
        """
        )
        return [record["article_id"] for record in result]
    except:
        return []


@retry(tries=100, delay=10)
def load_article_graph_from_csv() -> None:
    # load articles dataset
    articles = (
        pl.read_csv(ARTICLES_CSV_PATH, schema_overrides={"article_date": pl.Date})
        .filter(pl.col("article_text").is_not_null())
        .with_columns(pl.col("article_date").cast(pl.Date))
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
                    "url": row["article_url"],
                    "title": row["title"],
                    "year": row["article_date"].year,
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

            if a["date"]:
                # article_date = datetime.strptime(a['date'],'%Y-%m-%d').date()
                article_date = a["date"]
                # print(article_date)
                article_year = article_date.year
                year_cypher = f"""merge (y:{YEAR_NODE} {{year: date($date).year}})"""

                session.run(year_cypher, **a)
        LOGGER.info("Created Article nodes")

        LOGGER.info("Created Year nodes")

        published_rel_cypher = f"""MATCH (a:Article)
        WITH a, date(a.date).year AS articleYear
        MATCH (y:Year {{year: articleYear}})
        MERGE (a)-[:PUBLISHED_ON {{date: date(a.date)}}]->(y)"""

        session.run(published_rel_cypher)

        LOGGER.info("Created PUBLISHED_ON relationship")

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
                        "title": c["title"],
                        "url": c["url"],
                        "year": c["year"],
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

    with driver.session() as session:
        session.run(
            """MATCH (ac:ArticleChunk)<-[:HAS_CHUNK]-(a:Article)-[:PUBLISHED_ON]->(y:Year) SET ac.year = y.year"""
        )

        LOGGER.info("Set year on Article Chunk")


if __name__ == "__main__":
    load_article_graph_from_csv()
