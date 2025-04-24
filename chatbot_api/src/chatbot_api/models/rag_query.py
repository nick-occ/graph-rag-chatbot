from pydantic import BaseModel
from typing import List, Dict, Any


class QueryInput(BaseModel):
    text: str


class QueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: List[str]
    context: List[Dict[str, Any]]
