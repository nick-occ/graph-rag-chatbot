import os
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
QA_MODEL = os.getenv("QA_MODEL")

ARTICLE_CHUNK_NODE = "ArticleChunk"
ARTICLE_CHUNK_INDEX = "article_chunk_index"


# get the vector index
vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=ARTICLE_CHUNK_INDEX,
    node_label=ARTICLE_CHUNK_NODE,
    text_node_properties=["text"],
    embedding_node_property="embedding",
)

# create a retriever
retriever = vector_index.as_retriever(k=10)

# create and llm
llm = ChatOpenAI(model=QA_MODEL, temperature=0)

# create prompt
prompt = ChatPromptTemplate.from_template(
    """"Your job is to use the articles to answer questions related to the text in the articles. Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know. It is preferred to use more recent articles if 
there is conflicting information in more than one article.

<context>
{context}
</context>

Question: {input}
"""
)
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# create retrieval chain
retrieval_chain = create_retrieval_chain(
    retriever=retriever, combine_docs_chain=combine_docs_chain
)
