from enum import Enum
from functools import lru_cache
from typing import Union

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from config.vector_store import vector_store
from config.settings import Settings

load_dotenv()
app = FastAPI()


@lru_cache
def get_settings():
    return Settings()


@app.get("/")
def read_root():
    return "Hello World"


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


class ModelEnum(str, Enum):
    gemini_1dot5_flash = "gemini-1.5-flash"
    gemini_1dot5_pro = "gemini-1.5-pro"


class ChatInput(BaseModel):
    input: str
    model: ModelEnum = ModelEnum.gemini_1dot5_flash
    language: str = "English"


@app.post("/chat")
def chat(chat_input: ChatInput):
    llm = GoogleGenerativeAI(model=chat_input.model)
    syst_template = "Translate the following from {language} to italian"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", syst_template), ("user", "{text}")]
    )

    return llm.stream(
        prompt_template.invoke(
            {"language": chat_input.language, "text": chat_input.input}
        )
    )


class GetPDFContentInput(BaseModel):
    input: str


@app.get("/get-pdf-content-len")
def get_pdf_content_len(input: GetPDFContentInput):
    return vector_store.similarity_search(input.input)
