import json
from typing import Callable, List

import requests
from pydantic import BaseModel

from crc_api.crc import DEFAULT_BOUNDARY_STRING


class AskResponseProcessor(BaseModel):
    """
    Stateful processing of the multi-part response from the
    Conversation Retrieval Chain (CRC) endpoint.
    """

    completion_callback: Callable[[str], None]  # Called as CRC completion chunks arrive
    raw_response: str = ""  # Builds up raw response
    docs: List[str] = []  # Populated with docs once available
    docs_done: bool = False

    def process_chunk(self, chunk: str):
        self.raw_response += chunk
        if not self.docs_done and DEFAULT_BOUNDARY_STRING in self.raw_response:
            # we have received all docs now
            raw_docs, rest = tuple(
                map(str.strip, self.raw_response.split(DEFAULT_BOUNDARY_STRING))
            )
            raw_json = "[" + raw_docs.replace("}\n{", "},{") + "]"
            self.docs = [item["content"] for item in json.loads(raw_json)]
            self.docs_done = True

            # still need to process any content after the boundary
            chunk = rest.lstrip()

        if self.docs_done:
            self.completion_callback(chunk)


def get_token(base_url):
    '"" Obtain and return JWT token'
    response = requests.post(
        base_url + "/token",
        data={
            "client_id": "demo_client_id",
            "client_secret": "demo_client_secret",
        },
    )
    response.raise_for_status()

    return response.json()["access_token"]


def ask(
    conversation_id="123",
    question="Describe the most popular AI publications of 2021",
    base_url="http://127.0.0.1:8000",
):
    base_url = base_url.rstrip("/")

    token = get_token(base_url)

    response = requests.post(
        base_url + "/ask",
        headers={
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {token}",
        },
        json={
            "conversation_id": conversation_id,
            "question": question,
        },
        stream=True,
    )
    response.raise_for_status()

    def print_docs(docs):
        print("\n--- DOCS: ---\n")
        print("\n\n".join(docs))
        print("\n--- END OF DOCS ---\n\n")

    def completion_callback(token: str):
        # Print out tokens as they are received; flush is important to ensure there is no buffering
        print(token, end="", flush=True)

    proc = AskResponseProcessor(completion_callback=completion_callback)
    docs_printed = False
    for chunk in response.iter_content(chunk_size=None):  # handle data as it comes
        chunk_str = chunk.decode("utf-8")
        proc.process_chunk(chunk_str)

        if proc.docs and not docs_printed:
            print_docs(proc.docs)
            docs_printed = True


if __name__ == "__main__":
    import fire

    fire.Fire(ask)
