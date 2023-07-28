"""
Skeletal implementation of required persistence layer
"""

from collections import defaultdict

CLIENTS = {
    "demo_client_id": {
        "client_secret": "demo_client_secret",
    },
}

CONVERSATIONS = defaultdict(dict)


def get_client(client_id: str):
    return CLIENTS.get(client_id)


def get_conversation(conversation_id: str):
    return CONVERSATIONS[conversation_id]
