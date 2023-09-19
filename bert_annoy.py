import json
import shutil
from random import random

from sentence_transformers import SentenceTransformer, util
import os
import pickle
import time
import torch
from annoy import AnnoyIndex
from tqdm import tqdm
import sys

model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)

max_corpus_size = 100000

n_trees = 256  # Number of trees used for Annoy. More trees => better recall, worse run-time
embedding_size = 768  # Size of embeddings
top_k_hits = 10  # Output k hits

annoy_index_path = 'self-embeddings-{}-size-{}-annoy_index-trees-{}.ann'.format(model_name.replace('/', '_'),
                                                                                max_corpus_size, n_trees)
embedding_cache_path = 'self-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)
if os.path.exists(annoy_index_path): os.remove(annoy_index_path)
if os.path.exists(embedding_cache_path): os.remove(embedding_cache_path)


def main(current_data, target_data):
    if len(current_data) == 0:
        return
    global corpus_embeddings, annoy_index
    if not os.path.exists(embedding_cache_path):
        # Get all unique sentences from the file
        corpus_sentences = set()

        for content in current_data:
            corpus_sentences.add(content)
            if len(corpus_sentences) >= max_corpus_size:
                break

        corpus_sentences = list(corpus_sentences)
        # print("Encode the corpus. This might take a while")
        corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=False, convert_to_numpy=True)
        # print(corpus_embeddings)

        # print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)

    if not os.path.exists(annoy_index_path):
        # Create Annoy Index
        # print("Create Annoy index with {} trees. This can take some time.".format(n_trees))
        annoy_index = AnnoyIndex(embedding_size, 'angular')

        for i in range(len(corpus_embeddings)):
            annoy_index.add_item(i, corpus_embeddings[i])

        annoy_index.build(n_trees)
        annoy_index.save(annoy_index_path)

    corpus_embeddings = torch.Tensor(corpus_embeddings)

    ######### Search in the index ###########

    # print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

    inp_question = target_data

    start_time = time.time()
    question_embedding = model.encode(inp_question)

    corpus_ids, scores = annoy_index.get_nns_by_vector(question_embedding, top_k_hits, include_distances=True)
    hits = []
    for id, score in zip(corpus_ids, scores):
        hits.append({'corpus_id': id, 'score': 1 - ((score ** 2) / 2)})

    # end_time = time.time()
    # print("Input question:", inp_question)
    # print("Results (after {:.3f} seconds):".format(end_time-start_time))
    # for hit in hits[0:top_k_hits]:
    #     print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))
    thresholds = range(0, 100)
    for threshold in thresholds:
        if hits[0]['score'] >= (threshold / 100):
            store(f'result/bert_annoy/similar_{int(threshold)}.txt', target_data)
        else:
            break

def store(path, data):
    with open(path, 'a', encoding='utf8') as file:
        file.write(data + '\n')


def progress_bar(finish_tasks_number, tasks_number):
    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\rRate of Progress: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()


def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


def store_json(json_data, json_path):
    print(json_path, 'is storing...')
    with open(json_path, 'a', encoding='utf-8') as json_file:
        json_file.seek(0)
        json_file.truncate()
        json.dump(json_data, json_file, indent=4)
    json_file.close()


if __name__ == '__main__':
    setDir('result/bert_annoy')
    with open('test_bert_annoy.txt', 'r', encoding='utf-8') as file:
        target_data = eval(file.read())
    current_data = ["Conduct market research to determine potential new products or services that could be offered by the company."]
    for index in tqdm(range(len(target_data))):
        content = target_data[index]
        main(current_data, content)
        time.sleep(0.05)


