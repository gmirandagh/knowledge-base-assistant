import os
import json
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(PROJECT_ROOT, 'data', 'ground-truth-retrieval.csv')

if not os.path.exists(csv_path):
    print(f"Error: The ground truth file was not found at {csv_path}")
    print("Please make sure 'ground-truth-retrieval.csv' exists in the 'data' directory.")
    exit()

df = pd.read_csv(csv_path)
question = df.sample(n=1).iloc[0]['question']

print(f"Testing with question: \"{question}\"")
print("-" * 30)

url = "http://localhost:8000/ask"

data = {"question": question}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()

    print("API Response:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred while calling the API: {e}")
    print("Please ensure your Flask application (app.py) is running in another terminal.")