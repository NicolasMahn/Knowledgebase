import argparse
import math
import os
import shutil
import yaml
from langchain_community.document_loaders import PyPDFDirectoryLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from tqdm import tqdm
import ollama

from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

from get_data_path import TOPIC_PATH

CHROMA_PATH = f"{TOPIC_PATH}/chroma"
DATA_PATH = f"{TOPIC_PATH}/documents"
URL_MAPPING_FILE = f"{TOPIC_PATH}/url_mapping.yml"
BATCH_SIZE = 10  # Number of documents to process in each batch


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨  Clearing Database")
        clear_database()

    # Create (or update) the data store.
    print("⌛  Loading Data...")
    url_mapping = load_url_mapping()
    documents = load_documents(url_mapping)

    # Process documents in batches
    print(f"➗  Divided the {len(documents)} documents into batches of {BATCH_SIZE}")
    max_batches = math.ceil(len(documents) / BATCH_SIZE)
    white_text = "\033[37m"
    blue_bar = "\033[34m"
    reset_color = "\033[0m"

    # Custom bar format with color codes
    bar_format = f"{white_text}{{l_bar}}{blue_bar}{{bar}}{white_text}{{r_bar}}{reset_color}"
    with tqdm(total=max_batches, bar_format=bar_format, unit="batch", dynamic_ncols=True) as pbar:
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            chunks = split_documents(batch)
            add_to_chroma(chunks)
            pbar.update(1)


def load_url_mapping():
    with open(URL_MAPPING_FILE, 'r') as file:
        return yaml.safe_load(file).get('documents', {})


def load_documents(url_mapping):
    txt_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    csv_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    img_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(('.png', '.jpeg', '.svg'))]
    num_documents = len(txt_files)+len(csv_files)+len(img_files)

    white_text = "\033[37m"
    blue_bar = "\033[34m"
    reset_color = "\033[0m"

    # Custom bar format with color codes
    bar_format = f"{white_text}{{l_bar}}{blue_bar}{{bar}}{white_text}{{r_bar}}{reset_color}"
    with tqdm(total=num_documents, bar_format=bar_format, unit="batch", dynamic_ncols=True) as pbar:

        documents = []
        for txt_file in txt_files:
            documents.extend(load_txt_document(txt_file, url_mapping))
            pbar.update(1)

        for csv_file in csv_files:
            documents.extend(load_csv_document(csv_file, url_mapping))
            pbar.update(1)

        for img_file in img_files:
            documents.extend(load_img_document(img_file, url_mapping))
            pbar.update(1)

    return documents


def load_txt_document(file_path, url_mapping):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    doc_name = os.path.basename(file_path)
    url = url_mapping.get(doc_name, None)
    return [Document(page_content=content, metadata={"url": url, "doc_name": doc_name})]


def load_csv_document(file_path, url_mapping):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    doc_name = os.path.basename(file_path)
    url = url_mapping.get(doc_name, None)
    summary = generate_table_summary(content, get_context_for_url(url))
    return [Document(page_content=summary, metadata={"url": url, "doc_name": doc_name})]


def generate_table_summary(content, context):
    res = ollama.chat(
        model="mistral",
        messages=[
            {
                'role': 'user',
                'content': f'Describe this table: {content}'
                           f'---'
                           f'context: {context}',
            }
        ]
    )
    return res["message"]["content"]


def load_img_document(file_path, url_mapping):
    doc_name = os.path.basename(file_path)
    url = url_mapping.get(doc_name, None)
    summary = generate_img_summary(file_path, get_context_for_url(url))
    return [Document(page_content=summary, metadata={"url": url, "doc_name": doc_name, "type": "image"})]


def generate_img_summary(file_path, context):
    res = ollama.chat(
        model="llava",
        messages=[
            {
                'role': 'user',
                'content': f'Describe this image, given the context: {context}',
                'images': [file_path]
            }
        ]
    )
    return res["message"]["content"]


def get_context_for_url(url):
    context_file = os.path.join(DATA_PATH, "context_data.yaml")
    if not os.path.exists(context_file):
        # print("No context data available.")
        return None

    with open(context_file, 'r') as file:
        context_data = yaml.safe_load(file) or {'files': {}}

    result = {filename: data for filename, data in context_data['files'].items() if data['url'] == url}

    if not result:
        # print("No context data found for the specified URL.")
        return None

    return result


def load_pdf_documents(url_mapping):
    pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFDirectoryLoader(pdf_file)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc_name = os.path.basename(pdf_file)
            url = url_mapping.get(doc_name, None)
            doc.metadata["url"] = url
            doc.metadata["doc_name"] = doc_name
        documents.extend(loaded_docs)
    return documents


def load_html_documents(url_mapping):
    html_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.html')]
    documents = []
    for html_file in html_files:
        loader = UnstructuredHTMLLoader(html_file)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc_name = os.path.basename(html_file)
            url = url_mapping.get(doc_name, None)
            doc.metadata["url"] = url
            doc.metadata["doc_name"] = doc_name
        documents.extend(loaded_docs)
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    # if batch_info[0] == 1:
        # print(f"👉  Number of existing chunks in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist() <-- no longer necessary
    # else:
        # print(f"✅ No new documents to add in batch {batch_info[0]}")


def calculate_chunk_ids(chunks):
    for chunk in chunks:
        source = chunk.metadata.get("doc_name", None)
        page = chunk.metadata.get("page", None)
        url = chunk.metadata.get("url", None)
        chunk_index = chunks.index(chunk)
        chunk_id = f"{url}|{source}|{page}|{chunk_index}"
        chunk.metadata["id"] = chunk_id
    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
