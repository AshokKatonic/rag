import requests
import json

API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/v1/chat/completions"
TOKEN_ENDPOINT = f"{API_BASE_URL}/auth/token"

CLIENT_NAME = "chat_client"
JWT_TOKEN = None

def get_auth_token():

    global JWT_TOKEN
    
    if JWT_TOKEN:
        return JWT_TOKEN
    
    try:
        print("Generating authentication token...")
        token_request = {
            "client_name": CLIENT_NAME,
            "expires_hours": 24
        }
        
        response = requests.post(TOKEN_ENDPOINT, json=token_request, timeout=10)
        
        if response.status_code == 200:
            token_data = response.json()
            JWT_TOKEN = token_data["access_token"]
            print(f"Token generated successfully for client: {CLIENT_NAME}")
            return JWT_TOKEN
        else:
            print(f"Failed to generate token: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to token endpoint")
        return None
    except Exception as e:
        print(f"Error generating token: {e}")
        return None

def get_auth_headers():

    token = get_auth_token()
    if not token:
        raise Exception("Could not obtain authentication token")
    
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def non_streaming_chat():
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [{"role": "user", "content": "How does multitenancy work?"}],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False
    }

    try:
        print(f"Sending request to: {CHAT_ENDPOINT}")
        print(f"Question: {payload['messages'][0]['content']}\n")

        headers = get_auth_headers()
        response = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print("Response received:")
            print(f"ID: {data.get('id', 'N/A')}")
            print(f"Model: {data.get('model', 'N/A')}")
            print(f"Answer: {data['choices'][0]['message']['content']}")

            sources = data.get('sources', [])
            if sources:
                print(f"Sources ({data.get('source_count', 0)}):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            else:
                print("Sources: No sources found")
        elif response.status_code == 401:
            print("Authentication failed - token may be invalid or expired")
            print("Response: {response.text}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server")
        print("Make sure to run 'python api.py' first")
    except Exception as e:
        print(f"Error: {e}")

def streaming_chat():
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [{"role": "user", "content": "How does multitenancy work?"}],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": True
    }

    try:
        print(f"Sending streaming request to: {CHAT_ENDPOINT}")
        print(f"Question: {payload['messages'][0]['content']}\n")
        print("Streaming response:")
        print("-" * 30)

        headers = get_auth_headers()
        response = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, stream=True, timeout=30)

        if response.status_code == 200:
            full_response = ""
            sources = []

            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    if chunk.startswith("data: "):
                        data_str = chunk[6:]
                        if data_str.strip() == "[DONE]":
                            print("\nStream completed")
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if "sources" in delta:
                                sources = delta["sources"]
                            if content:
                                print(content, end="", flush=True)
                                full_response += content
                        except (json.JSONDecodeError, KeyError):
                            continue

            print("\n\nSources:")
            if sources:
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            else:
                print("  No sources found")
        elif response.status_code == 401:
            print("Authentication failed - token may be invalid or expired")
            print("Response: {response.text}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server")
        print("Make sure to run 'python api.py' first")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("RAG API Chat Completion with JWT Authentication")
    print("=" * 60)
    
    # Test token generation first
    token = get_auth_token()
    if not token:
        print("Cannot proceed without authentication token")
        return
    
    print(f"Using token for client: {CLIENT_NAME}")
    print()
    
    non_streaming_chat()
    print("\n" + "="*60 + "\n")
    streaming_chat()

if __name__ == "__main__":
    main()