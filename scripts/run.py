import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np

import mltle as mlt


GRAPH_FEATURES = 'g78'
GRAPH_TYPE = 'gin_eps0'
NORMALIZE = False
NORMALIZATION_TYPE = ''

start_time = time.time()


# the model from the HDF5 file
model = load_model('examples/graphdta-mltle-test-bindingdb/mltle/models/ResCNN1GIN5_pKd.hdf5')

# as from notebook:
data_path = 'examples/graphdta-mltle-test-bindingdb/data/data_pKd/'
order = ['pKi', 'pIC50', 'pKd', 'pEC50', 'is_active', 'qed', 'pH']

X_train = pd.read_csv(data_path + f"data_human_agg05_pKd_train.csv")[['smiles', 'target'] + order]
X_valid = pd.read_csv(data_path + f"data_human_agg05_pKd_valid.csv")[['smiles', 'target'] + order]
X_test = pd.read_csv(data_path + f"data_human_agg05_pKd_test.csv")[['smiles', 'target'] + order]

mapseq = mlt.datamap.MapSeq(drug_mode=GRAPH_FEATURES,
                            protein_mode='protein_3',
                            max_drug_len=100,
                            max_protein_len=1000,
                            graph_normalize=NORMALIZE,
                            graph_normalization_type=NORMALIZATION_TYPE)

drug_seqs = np.hstack((X_train['smiles'].unique(), X_valid['smiles'].unique(), X_test['smiles'].unique()))
protein_seqs = np.hstack((X_train['target'].unique(), X_valid['target'].unique(), X_test['target'].unique()))

map_drug, map_protein = mapseq.create_maps(drug_seqs=drug_seqs,
                                           protein_seqs=protein_seqs)

test_batch_size = mlt.training_utils.get_batch_size(X_test.shape[0])
print(test_batch_size)

test_gen = mlt.datagen.DataGen(X_test,
                               map_drug,
                               map_protein,
                               shuffle=False,
                               test_only=True, 
                               drug_graph_mode=True)

test_gen = test_gen.get_generator(test_batch_size)

# Perform inference
print("predicting")
prediction = model.predict(test_gen,
                           steps=X_test.shape[0] // test_batch_size,
                           verbose=1)
# Analyze the results

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

# 30/30 [==============================] - 4s 128ms/step
# 18.821038007736206 seconds overall.

# To test the results: (same as in notebook)
# for k, col in enumerate(order):
#     try:
#         plt.scatter(X_test[col], prediction[k], alpha=0.7, c='k')
#         plt.xlabel('true')
#         plt.ylabel('predicted')
#         y_true = X_test[col][X_test[col].notna()]
#         y_pred = prediction[k][X_test[col].notna()].ravel()
#         plt.title(col + ":\n" + mlt.training_utils.get_scores(y_true, y_pred))
#         plt.show()
#     except (ValueError, ZeroDivisionError) as e:
#         print(f'Empty set test set for: {col}, values sum = {X_test[col].sum()}')
#         print(f'or zero concordand pairs for the set of length 1, length of values set = {X_test[col].notna().sum()}')

