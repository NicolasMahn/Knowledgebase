import argparse
import os

import yaml
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

PROMPT_TEMPLATE = """
Answer the question based only on the following context. 
Indicate the source of each part of your answer in the format [source: source_id]:

{context}

---

Answer the question based on the above context: {question}
"""

WHITE = "\033[97m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config("config.yaml")
    data_topics = config['data_topics']
    default_topic = config['default_topic']

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    args = parser.parse_args()
    if args.debug:
        print(f"{ORANGE}⭕  DEBUG Mode Active{RESET}")

    selected_topic = args.topic if args.topic else default_topic
    topic_config = data_topics[selected_topic]
    topic_dir = topic_config['topic_dir']
    chroma_dir = f"{topic_dir}/chroma"
    data_dir = f"{topic_dir}/documents"

    query_text = args.query_text
    query_rag(query_text, chroma_dir, data_dir, args.debug)


def query_rag(query_text: str, chroma_dir: str, data_dir: str, debug: bool = False):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_texts = []
    sources = []
    for doc, _score in results:
        source_id = doc.metadata.get("id", "unknown")
        if doc.metadata.get("type") == "image":
            context_texts.append(f"[source: {source_id}]\n{doc.page_content}")
        else:
            raw_content = load_raw_document_content(doc.metadata["doc_name"], data_dir)
            context_texts.append(f"[source: {source_id}]\n{raw_content}")
        sources.append(source_id)

    context_text = "\n\n---\n\n".join(context_texts)
    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    if debug:
        print("Retrieved Summarize:\n", results)
        print("Context:\n", context_text)
        print("Sources:\n", sources)
        print("\n")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)


    print(f"{WHITE}{response_text}{RESET}")

    return response_text


def load_raw_document_content(doc_name: str, data_dir: str):
    file_path = os.path.join(data_dir, doc_name)
    if file_path.endswith('.txt') or file_path.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    return "Content not available"


if __name__ == "__main__":
    main()
