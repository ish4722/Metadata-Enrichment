import dataikuapi

DATAIKU_INSTANCE = "https://ai-sandbox-weu-des-prod-testing-e.app.weu-d1.delfi.slb-ds.com/"
DATAIKU_API_KEY = "dkuaps-XTfSOZO1UpULJaLlpGkmixRXyXq85P6J"
DATAIKU_PROJECT = "DIAGENTSERVER"

def get_token():
    client = dataikuapi.DSSClient(DATAIKU_INSTANCE, DATAIKU_API_KEY)
    project = client.get_project(DATAIKU_PROJECT)
    return project.get_variables()["local"]["token"]

