import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
import string
import unidecode
from nltk.stem.snowball import SnowballStemmer

def str_lower_rm_punctuation_rm_accents(text):
    """
    String cleaning:
        - Lower Case
        - Remove Accents
        - Remove Symbols

    Parameters
    ----------
    text : str
        String to clean

    Returns
    -------

    text_clean : str
        cleant string

    Examples
    --------

    ::

        x = 'Hola! Como estás?'
        str_lower_rm_punctuation_rm_accents(x)

        Out:
            'hola como estas'

        x = '汉字'
        x_ = str_lower_rm_punctuation_rm_accents(x)
        print(x_)

        Out:
            yi zi

    """

    if len(text) == 0:
        return ''

    # Puctuations to remove
    table = str.maketrans({
        key: None
        for key in string.punctuation
    })

    # Make sure to have a string
    text_cleant = str(text)

    # Remove Symbols
    text_cleant = text_cleant.translate(table)

    # Remove accents
    text_cleant = unidecode.unidecode(text_cleant)

    # Lower
    text_cleant = text_cleant.lower()

    return str(text_cleant)
    

def apply_stemmer(sentence):
    """
    """
    snowball = SnowballStemmer(language="english")

    sentence = str_lower_rm_punctuation_rm_accents(sentence)
    sentence = [
        snowball.stem(word)
        for word in sentence.split(' ')]

    sentence = ' '.join(sentence)

    return sentence

df['query'] = df['query'].apply(apply_stemmer)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
import os
import xmltodict

path_categ = os.path.join(
    '/workspace/datasets/product_data/categories/',
    'categories_0001_abcat0010000_to_pcmcat99300050000.xml')

with open(path_categ, 'r') as f:
    categ_data = f.read()

categ_data_dict = xmltodict.parse(categ_data)
category_dict = categ_data_dict['categories']['category']

parent_categ, parent_id = [], []
for categ_item in category_dict:
    categ_item_path_cat = categ_item['path']['category']

    if isinstance(categ_item_path_cat, list):
        parent = categ_item['path']['category'][0]
        childs = categ_item['path']['category'][1:]
    else:
        parent = categ_item['path']['category']
        childs = []
    
    for child in childs:
        parent_categ.append({
            'parent': parent['name'],
            'child': child['name']
        })
        parent_id.append({
            'parent': parent['id'],
            'child': child['id']
        })

        parent = child

parent_categ, parent_id = pd.DataFrame(parent_categ), pd.DataFrame(parent_id)
child_id_to_parent_id = parent_id.set_index('child')['parent'].to_dict()

# Remove items not in the categ. id set
def replace_categ(cat):
    """
    """
    if cat in categs_below_min_th.index:
        return child_id_to_parent_id.get(cat, np.nan)

    return cat

min_th = 10000

df['category_'] = df['category'].copy()
for _ in range(10):
    categ_vc = df['category_'].value_counts()
    categs_below_min_th = categ_vc[categ_vc < min_th]
    print(categs_below_min_th.shape)

    if categs_below_min_th.shape[0] == 0:
        break
    df['category_'] = df['category_'].apply(replace_categ)
df['category_'].nunique()
# Remov double spaces
import re

df['query'] = df['query'].apply(lambda x: re.sub('\\s+', ' ', x))

(df['category'] + '--' + df['query']).value_counts()


# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df.dropna(inplace=True)
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
