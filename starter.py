from langchain_openai import ChatOpenAI
from sauth_token import get_token


def remove_bearer(token):
    return token[len("Bearer "):] if token.startswith("Bearer ") else token

FM_HUB_BASE_URL =  'https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1'
PARTITION_ID =  'sandbox-weu-des-prod-testing-e'
TOKEN = remove_bearer(get_token())

model_params = {
    "model": "gpt-4o",
    "base_url": FM_HUB_BASE_URL,
    "api_key": TOKEN,
    "default_headers": {
        "slb-partition-id": PARTITION_ID,
    }
}

llm = ChatOpenAI(**model_params)
response = llm.invoke("Who is the Prime Minister of the India,give only name ")
print(response)
