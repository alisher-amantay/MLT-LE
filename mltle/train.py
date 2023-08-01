import tensorflow as tf
from transformers import TFBertModel, BertTokenizer, BertConfig
import re


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
model = TFBertModel.from_pretrained("Rostlab/prot_bert_bfd", from_pt=True)

def get_embeddings(protein_seqs):
    # Preprocess protein sequences
    processed_seqs = [re.sub(r"[UZOB]", "X", sequence) for sequence in protein_seqs]

    # Encode protein sequences
    ids = tokenizer.batch_encode_plus(processed_seqs, add_special_tokens=True, padding=True, return_tensors="tf")
    input_ids = ids['input_ids']
    attention_mask = ids['attention_mask']

    # Get embeddings
    embedding = model(input_ids)[0]

    mean_embedding = tf.reduce_mean(embedding, axis=1)

    return mean_embedding

# Example usage
# protein_seqs = ["AETCZAO", "SKTZP"]
# embeddings = get_embeddings(protein_seqs)
# print(embeddings)
