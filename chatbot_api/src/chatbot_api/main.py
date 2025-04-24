import json
from fastapi import FastAPI
from chatbot_api.agents.rag_agent import rag_agent_executor
from chatbot_api.models.rag_query import QueryInput, QueryOutput
from chatbot_api.utils.async_utils import async_retry

app = FastAPI(
    title="Urban Institute Articles Chatbot", description="Endpoints for a chatbot."
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This cas help when there are intermittent connection issues to external APIs.
    """

    return await rag_agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/article-rag-agent")
async def query_article_agent(query: QueryInput) -> QueryOutput:
    query_response = await invoke_agent_with_retry(query.text)

    context_docs = []
    for step in query_response.get("intermediate_steps", []):
        if isinstance(step, tuple) and len(step) == 2:
            output = step[1]
            if isinstance(output, dict) and "context" in output:
                context_docs.extend(output["context"])

    return QueryOutput(
        input=query.text,
        output=query_response["output"],
        intermediate_steps=[
            str(s) for s in query_response.get("intermediate_steps", [])
        ],
        context=context_docs,
    )
