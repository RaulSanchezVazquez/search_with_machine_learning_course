import pandas as pd

DATA_PATH = '/workspace/datasets/fasttext/shuffled_labeled_products.txt'
TRAIN_PATH = '/workspace/week2/train_pruned.txt'
TEST_PATH = '/workspace/week2/test_pruned.txt'

def filter_labels(size=100):
    """
    """
    data = pd.read_csv(DATA_PATH, header=None, sep='\t')

    labels = data[0].apply(lambda x: x.split(' ')[0])
    vc_labels = labels.value_counts()
    valid_labels = vc_labels[vc_labels >= size].index

    is_valid_row = labels.isin(valid_labels)

    return data[is_valid_row]

data_filterd = filter_labels(size=500)

train_data = data_filterd.sample(10000)
test_data = data_filterd[
    ~data_filterd.index.isin(train_data.index)
].sample(10000)

with open(TRAIN_PATH, 'w') as f:
    data_txt = '\n'.join(train_data[0].tolist())
    f.write(data_txt)

with open(TEST_PATH, 'w') as f:
    data_txt = '\n'.join(test_data[0].tolist())
    f.write(data_txt)
