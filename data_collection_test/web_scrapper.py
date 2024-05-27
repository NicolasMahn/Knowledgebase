import requests
import os
import warnings
import re
import yaml

from populate_database import DATA_PATH
from populate_database import URL_MAPPING_FILE


def scrape(url: str):
    # Send a GET request to the webpage
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        scrape_text(url, response.text)
    else:
        warnings.warn(f"Failed to retrieve {url}. \n Response: {response}", UserWarning)


def scrape_text(url: str, text: str):
    filename = url_to_filename(url)
    update_url_mapping(filename, url)
    html_file_path = f"../{DATA_PATH}/{filename}"

    os.makedirs(f"../{DATA_PATH}", exist_ok=True)
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(str(text))


def url_to_filename(url: str, pdf=False):
    # Replace illegal characters with underscores
    filename = re.sub(r'[<>:"./\\|?*]', '_', url[8:])
    if pdf:
        return f"{filename}.pdf"
    return f"{filename}.html"


def update_url_mapping(filename: str, url: str):
    if not os.path.exists(f"../{URL_MAPPING_FILE}"):
        url_mapping = {'documents': {}}
    else:
        with open(f"../{URL_MAPPING_FILE}", 'r') as file:
            url_mapping = yaml.safe_load(file) or {'documents': {}}

    url_mapping['documents'][filename] = url

    with open(f"../{URL_MAPPING_FILE}", 'w') as file:
        yaml.safe_dump(url_mapping, file)


if __name__ == "__main__":
    url = "https://frankaemika.github.io/docs/"  # Replace with your URL
    scrape(url)


def scrape_pdf(url, content):
    filename = url_to_filename(url, pdf=True)
    pdf_file_path = f"../{DATA_PATH}/{filename}"

    os.makedirs(f"../{DATA_PATH}", exist_ok=True)
    with open(pdf_file_path, 'wb') as file:
        file.write(content)

    update_url_mapping(filename, url)
    pass
