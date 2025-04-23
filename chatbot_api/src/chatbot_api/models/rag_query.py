from pydantic import BaseModel
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    text: str


class DocumentContext(BaseModel):
    page_content: str
    metadata: Dict[str, Optional[str]]


class QueryInput(BaseModel):
    text: str


class QueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: List[str]
