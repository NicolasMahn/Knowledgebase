import hashlib
import shutil
import time
import warnings
import os

import fitz
import pdfplumber as pdfplumber
import requests
import yaml
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin
from tqdm import tqdm
from PIL import Image
import io
import pandas as pd

from get_data_path import TOPIC_PATH
from populate_database import DATA_PATH
from populate_database import URL_MAPPING_FILE

HASHED_CONTENT_FILE = f"{TOPIC_PATH}/hashed_content.txt"
CONTEXT_FILE = f"{TOPIC_PATH}/context_data.yaml"


class WebCrawler:
    def __init__(self, start_urls, allowed_domains, max_depth=2, max_pages=100, reset=False):
        if reset:
            self.reset()
        self.start_urls = start_urls
        self.allowed_domains = allowed_domains
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.urls_to_visit = [(url, 0) for url in start_urls]
        self.pages_crawled = 0
        self.delay = 1
        self.retry_delay = 60
        self.content_hashes = self.load_hashes()

    @staticmethod
    def load_hashes():
        if os.path.exists(HASHED_CONTENT_FILE):
            with open(HASHED_CONTENT_FILE, 'r') as file:
                return set(line.strip() for line in file)
        return set()

    def save_hashes(self):
        with open(HASHED_CONTENT_FILE, 'w') as file:
            for content_hash in self.content_hashes:
                file.write(content_hash + '\n')

    def crawl(self):
        print(f"🕷️  Crawling has started")
        white_text = "\033[37m"
        green_bar = "\033[32m"
        reset_color = "\033[0m"

        # Custom bar format with color codes
        bar_format = f"{white_text}{{l_bar}}{green_bar}{{bar}}{white_text}{{r_bar}}{reset_color}"
        with tqdm(total=self.max_pages, bar_format=bar_format, unit="page", dynamic_ncols=True) as pbar:
            while self.urls_to_visit and self.pages_crawled < self.max_pages:
                url, depth = self.urls_to_visit.pop(0)
                if url in self.visited_urls or depth > self.max_depth:
                    continue
                self.visited_urls.add(url)
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        self.process_response(response, url, depth)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        self.parse_links(soup, url, depth)
                        self.pages_crawled += 1
                        pbar.update(1)
                    time.sleep(self.delay)  # Delay between requests
                    self.save_hashes()
                except requests.RequestException as e:
                    warnings.warn(f"Failed to retrieve {url}: {e}", UserWarning)
            self.save_hashes()

    def process_response(self, response, url, depth):
        if "surge protection" in response.text.lower():
            print(f"Surge protection triggered. Waiting for {self.retry_delay} seconds.")
            time.sleep(self.retry_delay)
            self.urls_to_visit.append((url, depth))  # Re-add URL to retry later
            return False
        content_hash = hashlib.md5(str(response.content).encode('utf-8')).hexdigest()
        if content_hash not in self.content_hashes:
            self.content_hashes.add(content_hash)
            if url.lower().endswith('.pdf'):
                self.process_pdf(url, response.content)
            else:
                self.process_html(url, response)

    def process_pdf(self, url, content):
        self.scrape_text_from_pdf(url, content)
        self.scrape_images_from_pdf(url, content)
        self.scrape_tables_from_pdf(url, content)

    def scrape_text_from_pdf(self, url, pdf_content):
        filename = self.url_to_filename(url, no_type=True)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() + "\n"
                page_filename = f"{filename}_page_{i+1}.txt"
                with open(f"{DATA_PATH}/{page_filename}", 'w', encoding='utf-8') as file:
                    file.write(page_text)
                self.update_url_mapping(page_filename, url)

    def scrape_images_from_pdf(self, url, pdf_content):
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        for i in range(len(pdf_document)):
            page = pdf_document.load_page(i)
            page_text = page.get_text("text")
            for j, img in enumerate(pdf_document.get_page_images(i)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                img_bytes = base_image["image"]
                img_format = self.get_image_format(img_bytes)
                if img_format:
                    img_filename = self.url_to_filename(url, file_format=f"{img_format.lower()}",
                                                        name_extension=f"_page_{i+1}_image_{j+1}")
                    self.process_images(img_bytes, url, img_filename, context=page_text)

    def scrape_tables_from_pdf(self, url, pdf_content):
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                page_text = page.extract_text()
                for j, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_filename = self.url_to_filename(url, no_type=True,
                                                              name_extension=f"_page_{i + 1}_table_{j + 1}.csv")
                        table_path = f"{DATA_PATH}/{table_filename}"
                        df.to_csv(table_path, index=False)
                        self.update_url_mapping(table_filename, url)
                        self.update_context_data(table_filename, url, page_text)

    def process_html(self, url, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        self.scrape_tables_from_html(url, soup)
        self.scrape_text_from_html(url, soup)
        self.scrape_images_from_html(url, soup)

    def scrape_text_from_html(self, url, soup):

        for table in soup.find_all('table'):
            table.decompose()
        text = soup.get_text(separator='\n')

        filename = self.url_to_filename(url, "txt")
        self.update_url_mapping(filename, url)
        html_file_path = f"{DATA_PATH}/{filename}"

        os.makedirs(DATA_PATH, exist_ok=True)
        with open(html_file_path, 'w', encoding='utf-8') as file:
            file.write(str(text))

    def scrape_tables_from_html(self, url, soup):
        os.makedirs(DATA_PATH, exist_ok=True)
        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            try:
                df = pd.read_html(io.StringIO(str(table)))[0]
                table_filename = self.url_to_filename(url, file_format="csv")
                table_path = f"{DATA_PATH}/{table_filename}"
                df.to_csv(table_path, index=False)
                surrounding_text = ' '.join(table.find_parent().stripped_strings)
                self.update_url_mapping(table_filename, url)
                self.update_context_data(table_filename, url, surrounding_text)
            except ValueError as e:
                warnings.warn(f"Failed to parse table at {url}: {e}", UserWarning)

    def scrape_images_from_html(self, base_url, soup):
        for img in soup.find_all('img'):
            context = f"alt: {img.get('alt', '')}, " \
                      f"caption: {img.find_next('figcaption').text if img.find_next('figcaption') else ''}, " \
                      f"surrounding_text: {' '.join(img.find_parent().stripped_strings)}"
            img_url = img.get('src')
            if not img_url:
                continue
            img_url = urljoin(base_url, img_url)
            try:
                img_response = requests.get(img_url)
                if img_response.status_code == 200:
                    self.process_images(img_response.content, img_url, context=context)
            except requests.RequestException as e:
                warnings.warn(f"Failed to retrieve image {img_url}: {e}", UserWarning)

    def process_images(self, img_data, img_url, img_filename=None, context=""):
        if "logo" not in img_url.lower() and "icon" not in img_url.lower():
            img_hash = hashlib.md5(str(img_data).encode('utf-8')).hexdigest()
            if img_hash not in self.content_hashes:
                self.content_hashes.add(img_hash)
                if img_filename is None:
                    img_format = self.get_image_format(img_data)
                    if not img_format:
                        img_format = "svg"
                    img_filename = self.url_to_filename(img_url, img_format.lower())

                with open(f"{DATA_PATH}/{img_filename}", 'wb') as img_file:
                    img_file.write(img_data)
                    self.update_url_mapping(img_filename, img_url)
                    self.update_context_data(img_filename, img_url, context)

    @staticmethod
    def get_image_format(image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.format
        except IOError:
            return None

    @staticmethod
    def url_to_filename(url: str, file_format="html", no_type=False, name_extension=""):
        # Replace illegal characters with underscores
        filename = re.sub(r'[<>:"./\\|?*]', '_', url[8:])
        if no_type:
            return f"{filename}{name_extension}"
        return f"{filename}{name_extension}.{file_format}"

    @staticmethod
    def update_url_mapping(filename: str, url: str):
        if not os.path.exists(URL_MAPPING_FILE):
            url_mapping = {'documents': {}}
        else:
            with open(URL_MAPPING_FILE, 'r') as file:
                url_mapping = yaml.safe_load(file) or {'documents': {}}

        url_mapping['documents'][filename] = url

        with open(URL_MAPPING_FILE, 'w') as file:
            yaml.safe_dump(url_mapping, file)

    def parse_links(self, soup, base_url, depth):
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not re.match(r'^https?://', href):
                href = urljoin(base_url, href)
            href_domain = urlparse(href).netloc
            if not any(allowed_domain in href_domain for allowed_domain in self.allowed_domains):
                continue
            if href not in self.visited_urls:
                self.urls_to_visit.append((href, depth + 1))
        return links

    @staticmethod
    def update_context_data(filename, url, context):
        if not os.path.exists(CONTEXT_FILE):
            context_data = {'files': {}}
        else:
            with open(CONTEXT_FILE, 'r') as file:
                context_data = yaml.safe_load(file) or {'files': {}}

        context_data['files'][filename] = {'url': url, 'context': context}

        with open(CONTEXT_FILE, 'w') as file:
            yaml.safe_dump(context_data, file)

    @staticmethod
    def reset():
        print("✨  Clearing Datastorage")
        if os.path.isfile(URL_MAPPING_FILE):
            os.remove(URL_MAPPING_FILE)

        if os.path.isfile(HASHED_CONTENT_FILE):
            os.remove(HASHED_CONTENT_FILE)

        if os.path.isdir(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        os.makedirs(DATA_PATH)

def main():
    allowed_domains = [
        "frankaemika.github.io",
        "github.com/frankaemika",
        "franka.de",
        "wiki.ros.org",
        "gazebosim.org",
        "devquantec.de/wp-content/uploads/2020/06/Datenblatt_Franka%20Emika%20Panda.pdf"]
    start_urls = [
        "https://devquantec.de/wp-content/uploads/2020/06/Datenblatt_Franka%20Emika%20Panda.pdf"
        "https://frankaemika.github.io/docs/",
        "https://github.com/frankaemika/",
        "https://www.franka.de/",
        "https://wiki.ros.org/",
        "https://gazebosim.org/docs"
    ]
    reset = True
    crawler = WebCrawler(start_urls=start_urls, allowed_domains=allowed_domains,
                         max_depth=2, max_pages=10, reset=reset)
    crawler.crawl()


if __name__ == "__main__":
    main()
