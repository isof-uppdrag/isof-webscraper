import queue
import threading
import argparse
from urllib.request import urlopen, Request
from urllib.parse import urlparse, urljoin, urlunparse
import logging
from bs4 import BeautifulSoup
import os
import hashlib
import json
from datetime import datetime, timezone

parser = argparse.ArgumentParser(description='Threaded web crawler')
parser.add_argument('-w', '--workers', required=False, type=int, default=4, help='Number of worker threads')
parser.add_argument('-l', '--limit', required=False, type=int, default=1000000, help='Number of links to crawl before stopping')
parser.add_argument('-j', '--jsonfile', required=True, help='JSON file with targets including webbadress and ort')
parser.add_argument('--skip-cached', action='store_true', help='Skip URLs that have already been crawled and saved')
commandArgs = parser.parse_args()

VALIDSCHEMES = ["https", "http"]

logging.basicConfig(level=logging.INFO, format='%(threadName)s: %(message)s')


class Crawler():
    def __init__(self, targets):
        self.linkStack = queue.Queue()
        self.checkedLinks = set()
        self.counter = 0
        self.running = True
        self.limit = commandArgs.limit
        self.workerNumber = commandArgs.workers
        self.lock = threading.Lock()
        self.domainToOrts = {}
        self.skipCached = commandArgs.skip_cached

        for target in targets:
            url = target.get("webbadress")
            ort = target.get("ort", "unknown")
            if url:
                parsed = urlparse(url)
                domain = parsed.netloc
                self.domainToOrts[domain] = ort
                filename = hashlib.md5(url.encode('utf-8')).hexdigest() + ".json"
                folder_path = os.path.join("data", ort)
                file_path = os.path.join(folder_path, filename)

                if os.path.exists(file_path):
                    if self.skipCached:
                        logging.info(f"Skipping already crawled: {url}")
                        continue
                    else:
                        try:
                            os.remove(file_path)
                            logging.info(f"Deleted cached content for: {url}")
                        except Exception as e:
                            logging.warning(f"Failed to delete cached file for {url}: {e}")

                self.linkStack.put(url)
                self.checkedLinks.add(url)

        os.makedirs("data", exist_ok=True)
        self.main()

    def save_content(self, domain, url, content):
        ort = self.domainToOrts.get(domain, domain.replace(":", "_").replace("/", "_"))
        folder_path = os.path.join("data", ort)
        os.makedirs(folder_path, exist_ok=True)
        filename = hashlib.md5(url.encode('utf-8')).hexdigest() + ".json"
        file_path = os.path.join(folder_path, filename)

        data = {
            "url": url,
            "date": datetime.now(timezone.utc).isoformat(),
            "content": content
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def read_html(self, inputHtml, currentLink):
        currentDomain = urlparse(currentLink).netloc
        soup = BeautifulSoup(inputHtml, "html.parser")
        fetchedUrls = [a.get("href") for a in soup.select("a") if a.get("href") is not None]

        for link in fetchedUrls:
            if '#' in link:
                continue  # Skip fragment links
            full_url = urljoin(currentLink, link)
            parsed_url = urlparse(full_url)._replace(fragment="")
            normalized_path = parsed_url.path.split(';')[0]  # Remove session ID
            cleaned_url = urlunparse(parsed_url._replace(path=normalized_path, query=""))
            if parsed_url.scheme in VALIDSCHEMES and parsed_url.netloc in self.domainToOrts:
                with self.lock:
                    if cleaned_url not in self.checkedLinks and self.counter < self.limit:
                        self.linkStack.put(cleaned_url)
                        self.checkedLinks.add(cleaned_url)

        page_text = " ".join([p.get_text() for p in soup.select("p")])
        self.save_content(currentDomain, currentLink, page_text)

        with self.lock:
            self.counter += 1

    def scrape_task(self):
        NON_HTML_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".svg", ".bmp", ".ico", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", ".tar", ".gz", ".mp4"]
        while self.running:
            try:
                currentLink = self.linkStack.get(timeout=3)
            except queue.Empty:
                break

            with self.lock:
                if self.counter >= self.limit:
                    self.running = False
                    break

            logging.info(f"Fetching {currentLink} - Count: {self.counter}")
            try:
                req = Request(currentLink, headers={"User-Agent": "Mozilla/5.0"})
                if any(currentLink.lower().endswith(ext) for ext in NON_HTML_EXTENSIONS):
                    logging.info(f"Skipping non-HTML or unsupported content: {currentLink}")
                    continue
                fetchedPage = urlopen(req).read()
                self.read_html(fetchedPage, currentLink)
            except Exception as e:
                logging.info(f"Could not fetch or parse {currentLink}: {e}. Attempting to continue with child links if possible.")

    def finished_reading(self):
        with open("read_links.txt", "w", encoding="utf-8") as outFile:
            for link in sorted(self.checkedLinks):
                outFile.write(link + "\n")

    def main(self):
        while self.running:
            active_threads = [t for t in threading.enumerate() if t.name.startswith("Thread-")]
            while len(active_threads) < self.workerNumber:
                new_thread = threading.Thread(target=self.scrape_task, name=f"Thread-{len(active_threads) + 1}")
                new_thread.start()
                active_threads.append(new_thread)

            for t in active_threads:
                t.join(timeout=1)

if __name__ == "__main__":
    with open(commandArgs.jsonfile, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    results = json_data.get("results", [])
    Crawler(results)

