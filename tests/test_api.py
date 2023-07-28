from fastapi import status


def test_protected_endpoint_with_valid_token(client, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}

    response = client.post(
        "/ask",
        headers=headers,
        json={
            "conversation_id": "123",
            "question": "Describe the most popular AI publications of 2021",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    assert '{"content":' in response.content.decode("utf-8")


def test_protected_endpoint_with_invalid_token(client):
    headers = {"Authorization": "Bearer invalid_token"}

    response = client.post(
        "/ask",
        headers=headers,
        json={
            "conversation_id": "123",
            "question": "Describe the most popular AI publications of 2021",
        },
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Invalid or expired token"}


def test_protected_endpoint_without_token(client):
    response = client.post("/ask")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Not authenticated"}
