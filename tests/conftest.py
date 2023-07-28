from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from crc_api.api import app
from crc_api.auth import create_access_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_token():
    payload = {"sub": "demo_client_id"}

    access_token = create_access_token(
        data=payload, expires_delta=timedelta(minutes=15)
    )
    return access_token
