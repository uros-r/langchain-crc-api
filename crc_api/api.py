from datetime import timedelta
from functools import lru_cache

from fastapi import Depends, FastAPI, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from crc_api import config, crc
from crc_api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    decode_access_token,
)
from crc_api.dao import get_client

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
app = FastAPI()


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """Get and memoize content retriever"""
    return crc.create_retriever(
        openai_api_key=config.OPENAI_API_KEY,
        persist_directory=config.AI_REPORT_CHROMADB_DIRECTORY,
    )


class Client(BaseModel):
    client_id: str
    client_secret: str


@app.post("/token")
def get_access_token(client_id: str = Form(...), client_secret: str = Form(...)):
    """Enables client to acquire a JWT access token upon presenting valid credentials"""

    # Retrieve client information from the data access module
    client_info = get_client(client_id)

    if client_info is not None and client_info["client_secret"] == client_secret:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": client_id}, expires_delta=access_token_expires
        )
        return JSONResponse(
            content={"access_token": access_token, "token_type": "bearer"}
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid client credentials",
    )


class QARequest(BaseModel):
    conversation_id: str
    question: str


@app.post("/ask", response_class=StreamingResponse)
async def generate_response(
    data: QARequest,
    retriever: VectorStoreRetriever = Depends(get_retriever),
    token: str = Depends(oauth2_scheme),
) -> StreamingResponse:
    """
    Invoke conversation retrieval chain, with memory for given conversation ID.
    Illustrates how a multi-part response can be returned, consisting of

    - source documents retrieved and used as few-shot examples
    - streamed response, token by token

    This is a proof of concept, addressing a specific need, and not meant to be
    API design best practice.
    """

    # Check if token present + valid
    # TODO: make this a reusable concern at endpoint level, not implementation logic
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    # Invoke the chain, stream the responses
    return StreamingResponse(
        crc.get_answer_async(
            openai_api_key=config.OPENAI_API_KEY,
            retriever=retriever,
            conversation_id=data.conversation_id,
            question=data.question,
            condense_question_prompt=config.CONDENSE_QUESTION_PROMPT,
            qa_prompt=config.QA_PROMPT,
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
