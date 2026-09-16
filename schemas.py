### the result of the file will hold all the schemas for the output we want i.e. we define the output of how our LLM will return. Structured output
from typing import List

from pydantic import BaseModel, Field


## Reflection agent: critique
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion): # inherits from AnswerQuestion meaning it has all the fields from AnswerQuestion and adds additional fields
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )