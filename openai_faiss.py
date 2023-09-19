import os
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader

os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"

def search_similarity(query, path):
    loader = TextLoader(path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n"])
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    docs_and_scores = db.similarity_search_with_score(query)
    for doc, score in docs_and_scores:
        print(f"similar_sentence: {doc.page_content}, score: {score}")
        print('-----------------------------------------------------')
    return docs_and_scores

def store_json(json_data, json_path):
    with open(json_path, 'a', encoding='utf-8') as json_file:
        json_file.seek(0)
        json_file.truncate()
        json.dump(json_data, json_file, indent=4)
    json_file.close()
    print(json_path, 'has been stored.')

if __name__ == '__main__':
    target = 'Assess the company\'s risk management strategies and recommend improvements as necessary.'
    docs_and_scores = search_similarity(target, 'test_openai_faiss.txt')
    sub_json = []
    for doc, score in docs_and_scores:
        sub_json.append(f"score: {score}, similar_sentence: {doc.page_content}")
    json_data = {
        'target': target,
        'similar_sentences': sub_json
    }
    store_json(json_data, './result/openai_faiss/result.json')
