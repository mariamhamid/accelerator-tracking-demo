# test_api.py
# This file contains unit tests for api.py

# TestClient allows us to call FastAPI endpoints
# without running a real server
from fastapi.testclient import TestClient

# Import the FastAPI app from api.py
from api import app
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api import app
# Create a test client for the FastAPI app
client = TestClient(app)


def test_health_endpoint():
    """
    This test checks that the /health endpoint:
    1. Responds with status code 200
    2. Returns the expected JSON response
    """

    # Send a GET request to /health
    response = client.get("/health")

    # Check that the request was successful
    assert response.status_code == 200

    # Check that the response body is correct
    assert response.headers.get("content-type","").startswith("application/json")
    assert response.json() == {"status": "API is running"}
