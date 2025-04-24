import json
import os
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain import hub
from ..chains.retreiver_chain import retrieval_chain

AGENT_MODEL = os.getenv("AGENT_MODEL")
agent_prompt = hub.pull("hwchase17/openai-functions-agent")


def agent_tool_wrapper(query: str) -> dict:
    return retrieval_chain.invoke({"input": query})


tools = [
    Tool(
        name="Articles",
        func=agent_tool_wrapper,
        description="""Useful in answering questions about the Urban Institute's 
        research that can be found in their articles page.
        """,
    )
]

chat_model = ChatOpenAI(model=AGENT_MODEL, temperature=0)

rag_agent = create_openai_functions_agent(
    llm=chat_model, prompt=agent_prompt, tools=tools
)

rag_agent_executor = AgentExecutor(
    agent=rag_agent, tools=tools, return_intermediate_steps=True, verbose=True
)
