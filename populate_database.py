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

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"


def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def main():
    config = load_config("config.yaml")
    data_topics = config['data_topics']
    default_topic = config['default_topic']

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    args = parser.parse_args()

    selected_topic = args.topic if args.topic else default_topic
    topic_config = data_topics[selected_topic]
    topic_dir = topic_config['topic_dir']

    populater = DatabaseManager(topic_dir, args.reset, args.debug)
    populater.save_data()


class DatabaseManager:
    def __init__(self, topic_dir: str, reset: bool = False, debug: bool = False):
        self.topic_dir = topic_dir
        self.chroma_dir = f"{topic_dir}/chroma"
        self.data_dir = f"{topic_dir}/documents"
        self.url_mapping_file = f"{topic_dir}/url_mapping.yml"
        self.context_file = f"{topic_dir}/context_data.yaml"

        self.context_data = self.open_context_data()

        self.debug = debug
        if debug:
            print(f"{ORANGE}⭕  DEBUG Mode Active{RESET}")
            print("Topic Dir:", topic_dir)
        if reset:
            print(f"{WHITE}✨  Clearing Database{RESET}")
            self.clear_database()
        self.url_mapping = self.load_url_mapping()

        self.db = Chroma(persist_directory=self.chroma_dir, embedding_function=get_embedding_function())

    def clear_database(self):
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)

    def save_data(self):
        txt_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        csv_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        img_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(
            ('.png', '.jpeg', '.svg'))]

        # Custom bar format with color codes
        bar_format_txt = f"{WHITE}⌛  Adding Text    {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"
        bar_format_csv = f"{WHITE}⌛  Adding Tables  {{l_bar}}{GREEN}{{bar}}{WHITE}{{r_bar}}{RESET}"
        bar_format_img = f"{WHITE}⌛  Adding Images  {{l_bar}}{ORANGE}{{bar}}{WHITE}{{r_bar}}{RESET}"

        ncols = shutil.get_terminal_size((80, 20)).columns - 10

        with tqdm(total=len(txt_files), bar_format=bar_format_txt, unit="document", ncols=ncols) as pbar:
            for txt_file in txt_files:
                meta_data = self.load_txt_metadata(txt_file)
                if self.unique(meta_data):
                    doc = self.generate_txt_summary(txt_file, meta_data)
                    self.add_to_chroma(doc)
                pbar.update(1)
        '''
        with tqdm(total=len(csv_files), bar_format=bar_format_csv, unit="document", ncols=ncols) as pbar:
            for csv_file in csv_files:
                meta_data = self.load_csv_metadata(csv_file)
                if self.unique(meta_data):
                    doc = self.generate_csv_summary(csv_file, meta_data)
                    self.add_to_chroma(doc)
                pbar.update(1)
        '''

        with tqdm(total=len(img_files), bar_format=bar_format_img, unit="document", ncols=ncols) as pbar:
            for img_file in img_files:
                meta_data = self.load_img_metadata(img_file)
                if self.unique(meta_data):
                    doc = self.generate_img_summary(img_file, meta_data)
                    self.add_to_chroma(doc)
                pbar.update(1)

    def load_url_mapping(self):
        with open(self.url_mapping_file, 'r') as file:
            return yaml.safe_load(file).get('documents', {})

    def unique(self, metadata: dict):
        # Load the existing database.
        self.db = Chroma(persist_directory=self.chroma_dir, embedding_function=get_embedding_function())

        # Calculate Page IDs.
        metadata = self.calculate_chunk_id(metadata)

        # Add or Update the documents.
        existing_items = self.db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])

        return metadata["id"] not in existing_ids

    @staticmethod
    def calculate_chunk_id(metadata: dict):
        source = metadata.get("doc_name", None)
        url = metadata.get("url", None)
        chunk_id = f"{url}|{source}"
        metadata["id"] = chunk_id
        return metadata

    def add_to_chroma(self, doc: Document):
        self.db.add_documents([doc], ids=[doc.metadata["id"]])

    def gather_context(self, file_path, base_url=None):
        context = [self.get_context_from_filename(file_path)]

        if base_url is not None:
            other_docs = self.filter_non_image_documents_for_url(base_url)

            for doc in other_docs:
                with open(doc, 'r', encoding='utf-8') as file:
                    content = file.read()
                context.append(content)
        return "\n".join(context)

    def get_context_from_filename(self, filename):
        basename = os.path.basename(filename)

        if basename in self.context_data['files']:
            return self.context_data['files'][basename].get('context', None)
        return None

    def filter_non_image_documents_for_url(self, specific_url):
        file_names = []
        for file_name, url in self.url_mapping.items():
            if specific_url in url and not url.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg')):
                file_names.append(file_name)
        return file_names

    def load_txt_metadata(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        doc_name = os.path.basename(file_path)
        url = self.url_mapping.get(doc_name, None)
        lines = content.split('\n')
        has_filename = len(lines) > 0 and lines[0].startswith("Filename:")
        has_branch = len(lines) > 2 and lines[2].startswith("Branch:")
        has_code_marker = "```" in content
        if has_filename and has_branch and has_code_marker:
            _type = "code"
        else:
            _type = "text"
        return {"url": url, "doc_name": doc_name, "type": _type}

    def generate_txt_summary(self, file_path: str, metadata: dict):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        res = ollama.chat(
            model="mistral",
            messages=[
                {
                    'role': 'user',
                    'content': f'Summarize this text in less then 800 tokens and put it into context:'
                               f'\n{content}'
                }
            ]
        )
        summary = res["message"]["content"]

        if self.debug:
            print()
            print("Metadata:\n", metadata)
            print("Content:\n", content)
            print("Summary:\n", summary)
        return Document(page_content=summary, metadata=metadata)

    def load_csv_metadata(self, file_path):
        doc_name = os.path.basename(file_path)
        url = self.url_mapping.get(doc_name, None)
        return {"url": url, "doc_name": doc_name, "type": "table"}

    def generate_csv_summary(self, file_path: str, metadata: dict):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        context = self.gather_context(file_path)

        res = ollama.chat(
            model="mistral",
            messages=[
                {
                    'role': 'user',
                    'content': f'Summarize this table in less then 800 tokens and put it into context:'
                               f'\n{content}'
                               f'\n\n---\n\n'
                               f'context: {context}',
                }
            ]
        )
        summary = res["message"]["content"]

        if self.debug:
            print()
            print("Metadata:\n", metadata)
            print("Content:\n", content)
            print("Context:\n", context)
            print("Summary:\n", summary)
        return Document(page_content=summary, metadata=metadata)

    def load_img_metadata(self, file_path):
        doc_name = os.path.basename(file_path)
        url = self.url_mapping.get(doc_name, None)
        metadata = {"url": url, "doc_name": doc_name, "type": "image"}
        base_url = self.get_base_url_from_filename(doc_name)
        if base_url:
            metadata["base_url"] = base_url
        return metadata

    def generate_img_summary(self, file_path: str, metadata: dict):
        if "base_url" in metadata.keys():
            context = self.gather_context(file_path, metadata["base_url"])
        else:
            context = self.gather_context(file_path)

        res = ollama.chat(
            model="llava",
            messages=[
                {
                    'role': 'user',
                    'content': f'Describe this image in less then 800 tokens and put it into context: {context}',
                    'images': [file_path]
                }
            ]
        )
        summary = res["message"]["content"]

        if self.debug:
            print()
            print("Metadata:\n", metadata)
            print("Context:\n", context)
            print("Summary:\n", summary)
        return Document(page_content=summary, metadata=metadata)

    def get_base_url_from_filename(self, filename):
        basename = os.path.basename(filename)

        if basename in self.context_data['files']:
            return self.context_data['files'][basename].get('base_url', None)
        return None

    def open_context_data(self):
        with open(self.context_file, 'r') as file:
            context_data = yaml.safe_load(file) or {'files': {}}
        return context_data


if __name__ == "__main__":
    main()