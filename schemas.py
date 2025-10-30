from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """ Source schema for the agent to use when searching the web """

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """ Agent response schema for the agent to use when answering the question """
    
    answer: str = Field(description="The agent's answer to the question")
    sources: List[Source] = Field(
        default_factory=list, description="The sources used to answer the question")
