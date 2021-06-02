from rich.progress import track
import requests 
import time

string_api_url = "https://string-db.org/api"
output_format = "tsv-no-header"
method = "interaction_partners"

request_url = "/".join([string_api_url, output_format, method])

results = open('links.txt', 'w')

genes = [l.strip() for l in open('valid_string.txt')]

for gene in track(genes):
    params = {
        "identifiers" : gene, 
        "limit" : 5000,
        "caller_identity" : "www.awesome_app.org" 
    }

    response = requests.post(request_url, data=params)

    results.write(response.text)
    time.sleep(0.5)