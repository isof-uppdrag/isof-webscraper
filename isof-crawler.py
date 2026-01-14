import requests
import hashlib
import json
import time
import os
import fasttext
import argparse
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from collections import deque
from datetime import datetime, timezone

from langclassifier import fin_fit_disambiguation

#############
# Paths and config
#############

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory for files
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Folder to which the output (corpus in json format) and the logs should be written
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create folder

LANGCODES_PATH = os.path.join(BASE_DIR, "json", "langcodes-merged.json")  # File containing the lookup table for consistent ISO 639-3 codes
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scraped-corpus.json")  # Output (corpus) file
LOG_FILE = os.path.join(OUTPUT_DIR, "crawler.log")  # Log file

#############
# Logging the crawling process
#############

log_file = open(LOG_FILE, "w", encoding="utf-8")
log_lock = threading.Lock()

def log(msg):
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    full = f"[{ts}] {msg}"
    with log_lock:
        print(full)  # Print for user feedback
        log_file.write(full + "\n")
        log_file.flush()

#############
# Global variables
#############

#
start_urls = []  # Starting point
urlcat_lookup = {}  # Metadata for type of URL (kommun/region/myndighet)
allowed_domains = set()  # Making sure that only child links within the seed page are explored (no escpaing)
robots_cache = {}  # For robots.txt

# Language related
unique_lang_tags = set()
lang_counts = {}

visited = set()

state_lock = threading.Lock()

#############
# Helper functions
#############

def fasttext_predict(model, text):
    '''
    Helper function to predict language using the off-the-shelf Fasttext model
    '''
    label, prob = model.predict(text, k=1)
    return label[0].replace("__label__", ""), prob.item()

def to_iso3(lang_code):
    '''
    Normalize Fasttext predicted language code to ISO 639-3.
    Falls back to original value if unknown.
    '''
    return fasttext_to_iso3.get(lang_code, lang_code)

# Making sure only allowed content is scraped
def load_robots(netloc, scheme):
    robots_url = f"{scheme}://{netloc}/robots.txt"
    try:
        r = requests.get(
            robots_url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if r.status_code != 200:  # Status code 200 means successful status
            log(f"robots.txt returned an error: {robots_url} ({r.status_code})")
            return None

        rp = RobotFileParser()
        rp.parse(r.text.splitlines())
        return rp

    except Exception as e:
        log(f"Could not fetch robots.txt from {robots_url}: {e}")
        return None

# Is scraping allowed as per robots.txt or not
def is_allowed(url, user_agent="*"):
    '''
    <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
    '''
    parsed = urlparse(url)
    netloc = parsed.netloc  # Netloc = website + domain
    scheme = parsed.scheme  # Scheme = http/https

    if netloc not in robots_cache:
        robots_cache[netloc] = load_robots(netloc, scheme)
    rp = robots_cache.get(netloc)

    return True if rp is None else rp.can_fetch(user_agent, url)

def make_unique_id(category, url):
    '''
    Create unique ID as metadata for each scraped text
    '''
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    raw = f"{category}|{base}".lower().strip()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def clean_text(soup):
    for tag in soup(["nav", "header", "footer", "script", "style", "img"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

#############
# Page parser
#############

def parse_page(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Extracting lang code from HTML code
    html_tag = soup.find("html")
    lang_html = html_tag.get("lang", "").lower() if html_tag else ""

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = clean_text(soup)

    # Strip text from title if text starts with title
    if title and text.startswith(title):
        text = text[len(title):].lstrip(" --: ")

    length = len(text)

    try:
        # Run off-the-shelf Fasttext model language prediction on scraped text
        fast_lang, conf = fasttext_predict(FAST_MODEL_ALL, text)
    except Exception:
        fast_lang, conf = "unknown", 0.0

    final_prediction = to_iso3(fast_lang)  # Make sure that the 639-3 format is used
    classification_type = "Fasttext off-the-shelf"  # Classification_type is used as metadata in the output

    # Metadata for features used for the rule-based prediction are not populated when classification was done with off-the-shelf Fasttext
    rule_features = {
        "count_d": None,
        "rel_freq_d": None,
        "count_h": None,
        "rel_freq_h": None,
        "count_ette": None,
        "count_oon": None,
        "count_mie": None,
        "count_sie": None,
    }

    # Disambiguating Finnish and Meänkieli -- assuming that the off-the-shelf model predicted "fin" for both
    if final_prediction == "fin":

        # Rule-based
        if DISAMBIGUATION_TYPE == "rule":
            rule_result = fin_fit_disambiguation(text)
            final_prediction = rule_result["final_prediction"]
            for k in rule_features:
                rule_features[k] = rule_result[k]
            classification_type = "Rule-based"

        # Disambiguator model
        elif DISAMBIGUATION_TYPE == "model":
            try:
                finfit_label, _ = fasttext_predict(FAST_MODEL_FINFIT, text)
                final_prediction = to_iso3(finfit_label)
            except Exception:
                final_prediction = "fin"
            classification_type = "Fasttext trained disambiguator model"

    # Romani - language prediction currently based solely on HTML tags
    if lang_html in {"rmf", "rmn", "rmy", "rmu", "rom"}:
        final_prediction = "rom"
        classification_type = "HTML overwrite"

    # Sami - language prediction currently based solely on HTML tags
    if lang_html in {"smi", "smj", "sma", "sme", "se"} or "samegi" in url:
        final_prediction = "smi"
        classification_type = "HTML overwrite"

    # This could help in case Yiddish is written with latin alphabet
    if lang_html == "yi" and final_prediction == "deu":
        final_prediction = "yid"
        classification_type = "HTML overwrite"

    with state_lock:
        # Number of texts per predicted language
        lang_counts[final_prediction] = lang_counts.get(final_prediction, 0) + 1

        # Number of unique languages predicted
        unique_lang_tags.add(final_prediction)

        log(f"Languages seen so far: {sorted(unique_lang_tags)}")
        log(f"Counts per language: {lang_counts}")

    # Metadata
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    category = urlcat_lookup.get(url)
    uid = make_unique_id(category, url)

    # Storing (internal) child links to add to queue later
    links = []
    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"]).split("#")[0]
        if urlparse(link).netloc in allowed_domains:
            links.append(link)

    # Output with metadata; links are not written to the output for smaller file size, only stored in the back-end
    return {
        "uid": uid,
        "url": url,
        "category": category,
        "lang-url-tag": lang_html or "unknown",
        "length": length,
        "lang-fasttext-identified": to_iso3(fast_lang),
        "lang-fasttext-confidence": conf,
        "classification_type": classification_type,
        **rule_features,
        "final_prediction": final_prediction,
        "date": timestamp,
        "title": title,
        "text": text
    }, links

#############
# Worker (threaded)
#############

def crawl_one(url, outfile, seed_set, seeds_remaining):
    global seeds_processed

    if not is_allowed(url, "Mozilla/5.0"):
        log(f"Skipping {url} due to robots.txt")
        return []
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return
    except requests.exceptions.SSLError as ssl_err:
        log(f"SSL error while crawling {url}: {ssl_err}")
        return []
    except requests.exceptions.RequestException as e:
        log(f"Failed to crawl {url}: {e}")
        return []

    log(f"Crawling {url}")

    page_data, child_links = parse_page(r.text, url)

    with state_lock:
        json.dump(page_data, outfile, ensure_ascii=False, indent=4)
        outfile.write("\n")

        if THREADING_ENABLED and url in seeds_remaining:
            seeds_remaining.remove(url)
            if not seeds_remaining and not seeds_processed:
                seeds_processed = True
                log("*** All seed URLs processed; now crawling child links ***")

    time.sleep(0.2)

    return child_links

#############
# Main
#############

def main():
    global FAST_MODEL_ALL, FAST_MODEL_FINFIT
    global DISAMBIGUATION_TYPE, fasttext_to_iso3
    global THREADING_ENABLED, seeds_processed

    log("Starting crawler")

    parser = argparse.ArgumentParser(description="Web crawler")
    #parser.add_argument("-a", "--fast-model-all", required=True, help="Off-the-shelf Fasttext model")
    parser.add_argument("-f", "--finfit-model", help="Trained model for Finnish-Meänkieli disambiguation; Not required in case disambiguation-type = rule")
    parser.add_argument("-i", "--input", required=True, help="Input file containing target URLs")
    parser.add_argument(
        "-d", "--disambiguation-type",
        required=True,
        choices=["rule", "model"],
        help="Disambiguation type (rule or model) for Finnish/Meänkieli")
    
    # Threading
    parser.add_argument("-t", "--threading", action="store_true", help="Enable threaded crawling")
    parser.add_argument("-w", "--max_workers", type=int, default=4, help="Number of workers for threaded crawling (default=4)")

    args = parser.parse_args()

    THREADING_ENABLED = args.threading  # True or false

    # User feedback for missing argument
    if args.disambiguation_type == "model" and not args.fast_model_finfit:
        parser.error("--finfit-model is required when --disambiguation-type=model")

    DISAMBIGUATION_TYPE = args.disambiguation_type
    #FAST_MODEL_ALL = fasttext.load_model(args.fast_model_all)
    FAST_MODEL_ALL = fasttext.load_model(os.path.join(BASE_DIR, "models", "lid.176.ftz"))
    FAST_MODEL_FINFIT = None
    if DISAMBIGUATION_TYPE == "model":
        FAST_MODEL_FINFIT = fasttext.load_model(args.fast_model_finfit)

    log("Fasttext model(s) loaded")

    with open(LANGCODES_PATH, "r", encoding="utf-8") as f:
        langcodes = json.load(f)

    fasttext_to_iso3 = {
        item["code-fasttext"]: item["code-iso"]
        for item in langcodes
    }

    # Opening and starting to process input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for category, block in data.items():
        urlcat = block.get("domain", category)  # Get metadata "Category"
        for entry in block.get("targets", []):
            url = entry.get("url")
            if not url:
                continue
            parsed = urlparse(url)
            if not parsed.scheme:
                url = "http://" + url
            start_urls.append(url)
            urlcat_lookup[url] = urlcat
            allowed_domains.add(urlparse(url).netloc)

    start_urls[:] = list(set(start_urls))
    log(f"Loaded {len(start_urls)} seed URLs")

    seed_set = set(start_urls)
    seeds_remaining = set(start_urls)
    seeds_processed = False

    visited = set()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        # Without threading
        if not THREADING_ENABLED:
            log(f"Crawler running with no threading")
            queue = deque(start_urls)
            while queue:
                url = queue.popleft()  # Assign URL from queue

                if not seeds_processed and all(s in visited for s in seed_set):
                    seeds_processed = True
                    log("*** All seed URLs processed; now crawling child links ***")

                if url in visited:
                    continue

                visited.add(url)

                child_links = crawl_one(url, outfile, seed_set, seeds_remaining)
                for link in child_links:
                    if link not in visited:
                        urlcat_lookup[link] = urlcat_lookup.get(url)
                        queue.append(link)

                log(f"Queue length: {len(queue)}")
                log("")

        # With threading
        else:
            log(f"Threading enabled with {args.max_workers} workers")
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(crawl_one, url, outfile, seed_set, seeds_remaining): url for url in start_urls}

                for future in as_completed(futures):
                    child_links = future.result()
                    for link in child_links:
                        if link not in visited:
                            visited.add(link)
                            executor.submit(crawl_one, link, outfile, seed_set, seeds_remaining)

    log("Crawl finished")
    log_file.close()

# Run only if run as a main program
if __name__ == "__main__":
    main()