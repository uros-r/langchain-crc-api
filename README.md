# FastAPI Endpoint for Langchain Conversation Retrieval Chain

## Brief

Create a proof-of-concept for a secure endpoint to generate a response from Langchain's conversational retrieval chain.
Use the following stack:

- FastAPI
- Langchain
- OpenAI
- Chroma (easily swappable)

The endpoint should:

- be secured with OAuth2 and JWT tokens
- return the source documents as JSON payload
- stream the completion response in real time
- support multiple conversations, identifiable via an ID and backed by ConversationBufferWindowMemory
- enable easy modification of default prompts (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
- be easy to containerise + configure via environment

## Implementation

This is a standalone Python program that demonstrates how the above requirements can be implemented.

API design, effectively representing a multi-part combination of a) JSON documents and b)
a text stream, was a requirement for a fairly niche use case. In practice, this makes clients
and processing tricker, and a separation of streaming and document retrieval would normally be
a better option.

The implementation consists of:

- `api.py` provides the API
- `auth.py` implements JWT authentication helpers
- `client.py` is a Python client for the endpoint, showing how a response can be consumed and processed
- `config.py` provides point of access for external configuration
- `crc.py` provides the Conversation Retrieval Chain functionality on top of documents and retriever (vector store) management
- `dao.py` contains a skeletal implementation of a client cred store and a conversation store, a production implementation would replace these
- `tests/` contains a pytest test suite

Supporting notebooks:

- `notebooks/init_chroma_vectorstore.ipynb` showing howw to initialise the vector store
- `notebooks/crc.ipynb`, a notebook for playing with the CRC interface

## Running the API server

1. Load the desired doc(s) into the vector store by modifying `notebooks/init_chroma_vectorstore.ipynb`
1. Run the API server (in dev mode, with reload enabled):

```bash
poetry run python crc_api/api.py
```

## Testing the endpoint

### From the command line

1. Get a token

   ```bash
   curl -X 'POST'  \
     -H 'Content-Type: application/x-www-form-urlencoded' \
     -d 'client_id=demo_client_id&client_secret=demo_client_secret'
     http://127.0.0.1:8000/token
   ```

1. Invoke the completion endpoint, replacing the `<TOKEN>` with the one from the previous step:

   ```bash
   curl --no-buffer -X POST -H 'accept: text/event-stream' \
     -H 'Content-Type: application/json' \
     -H 'Authorization: Bearer <TOKEN>' \
     -d '{"conversation_id": "123", "question": "What is the total number of AI publications in 2021?"}' \
     http://localhost:8000/ask
   ```

### Using the Python client

The client performs similar steps to curl - gets a token and makes a request.
It then unpicks the response to extract out the docs and print completion tokens as they arrive.

```
poetry run python crc_api/client.py
```
