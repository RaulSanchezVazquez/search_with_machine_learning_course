"""
pyenv activate search_with_ml

python index_products.py $REDUCE -s "$DATASETS_DIR/product_data/products"
open /workspace/ltr_output/ltr_model_importance.png

python week1/utilities/build_ltr.py --xgb_test /workspace/ltr_output/test.csv --train_file /workspace/ltr_output/train.csv --output_dir /workspace/ltr_output --xgb_test_num_queries 100 --xgb_main_query 0 --xgb_rescore_query_weight 2 && python week1/utilities/build_ltr.py --analyze --output_dir /workspace/ltr_output
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


> import nltk
> nltk.download('punkt')
> nltk.download('averaged_perceptron_tagger')
> nltk.download('maxent_ne_chunker')
> nltk.download('words')
str = "brand new iphone 13"
nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(str)))
