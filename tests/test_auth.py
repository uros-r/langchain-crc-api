from datetime import timedelta

import jwt
import pytest

from crc_api.auth import create_access_token, decode_access_token
from crc_api.config import JWT_SECRET_KEY

ALGORITHM = "HS256"


@pytest.fixture
def valid_payload():
    return {"sub": "test_user"}


@pytest.fixture
def valid_token(valid_payload):
    return jwt.encode(valid_payload, JWT_SECRET_KEY, algorithm=ALGORITHM)


@pytest.fixture
def invalid_token():
    return "invalid_token"


@pytest.fixture
def expired_token(valid_payload):
    expired_payload = valid_payload.copy()
    expired_payload["exp"] = 0  # Set expiration time to the past (expired token)
    return jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=ALGORITHM)


def test_create_access_token():
    data = {"sub": "test_user"}

    access_token = create_access_token(data, expires_delta=timedelta(minutes=15))

    # Verify the token is not empty
    assert access_token is not None

    # Verify the token can be decoded successfully
    decoded_token = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded_token.get("sub") == "test_user"


def test_decode_access_token_with_valid_token(valid_token):
    payload = decode_access_token(valid_token)

    # Verify the payload is correct
    assert payload is not None
    assert payload.get("sub") == "test_user"


def test_decode_access_token_with_invalid_token(invalid_token):
    payload = decode_access_token(invalid_token)

    # Verify the payload is None for invalid tokens
    assert payload is None


def test_decode_access_token_with_expired_token(expired_token):
    payload = decode_access_token(expired_token)

    # Verify the payload is None for expired tokens
    assert payload is None
