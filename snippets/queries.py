"""
pyenv activate search_with_ml

open /workspace/ltr_output/ltr_model_importance.png
"""
import json
import sys
import tempfile
from urllib.parse import urljoin

import requests
import xgboost as xgb
from opensearchpy import OpenSearch
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_tree

host = 'localhost'
port = 9200
base_url = "https://{}:{}/".format(host, port)

client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_compress=True,
    http_auth=('admin', 'admin'),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False)

# Get all indices
indices_metadata = client.indices.get_alias("*")
for index, index_meta in indices_metadata.items():
    if '.' == index[0]:
        continue
    print(index)

# Get sample of docs in index
index_name = 'bbuy_queries'
index_name = 'bbuy_products'
q_res = client.search(
    body={},
    index=index_name)
q_res['hits']['hits'][0]
