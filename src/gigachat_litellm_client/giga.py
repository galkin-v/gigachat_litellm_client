from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    role: Role
    content: str


class Dialogue(BaseModel):
    text: str


class GigaChatPayload(BaseModel):
    messages: List[Message]
    model: str
    temperature: float
    max_tokens: int
    profanity_check: bool


class Choice(BaseModel):
    message: Message
    index: int
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    precached_prompt_tokens: int
    prom_speed: Optional[float] = None


class GigaChatResponse(BaseModel):
    choices: List[Choice]
    created: int
    model: str
    object: str
    usage: Usage

    @property
    def content(self):
        return self.choices[0].message.content
