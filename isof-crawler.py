import requests
import hashlib
import json
import time
import os
import fasttext
import argparse
import threading
import re

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from collections import deque
from datetime import datetime, timezone
from collections import Counter

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

log_file = open(LOG_FILE, "a", encoding="utf-8")
log_lock = threading.Lock()

def log(msg):
    ts = datetime.now().replace(microsecond=0).isoformat()  # Uses system local time
    full = f"[{ts}] {msg}"
    with log_lock:
        print(full)  # Print for user feedback
        log_file.write(full + "\n")
        log_file.flush()

#############
# Global variables
#############

# Crawl related
start_urls = []  # Starting point
urlcat_lookup = {}  # Metadata for type of URL (kommun/region/myndighet/radio)
allowed_domains = set()  # Making sure that only child links within the seed page are explored (no escpaing)
robots_cache = {}  # For robots.txt
seen_text_hashes = set()
visited = set()
seeds_done = set()
seeds_processed = False

# Language related
unique_lang_tags = set()
lang_counts = {}

# Threading related
state_lock = threading.Lock()  # Only one thread is allowed to change shared memory at a time
file_lock = threading.Lock()  # Only one thread is allowed to write to output file at a time

#############
# Coverage tracking
#############

crawl_stats = {
    "discovered": set(),
    "crawled": set(),
    "failed": {}
}

#############
# Helper functions
#############

def normalize_netloc(netloc):
    '''
    Helps with avoiding potential duplicate visits
    '''
    return netloc.lower().lstrip("www.")

def classify_failure(exc):
    '''
    Categorizing web request failures - can be useful for debugging
    '''
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, requests.exceptions.SSLError):
        return "ssl_error"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection_error"
    return "request_error"

def fasttext_predict(model, text):
    '''
    Helper function to predict language using the off-the-shelf Fasttext model
    '''
    label, prob = model.predict(text, k=1)
    return label[0].replace("__label__", ""), prob.item()

def split_sentences(text):
    '''
    Split input text into sentences using regex
    '''
    return re.split(r'(?<=[.!?])\s+', text)  # Naive sentence split, potentially can be replaced with something more sophisticated like NLTK

def predict_language_sentence_level(text, model):
    '''
    Predict language of a whole text/document on the sentence level. 
    Returns a final, aggregated prediction based on majority vote,
    the average confidence for the final prediction, and a dictionary of sentence counts per language.
    '''
    sentences = split_sentences(text)

    predictions = []
    confidences = []

    for sent in sentences:
        if len(sent.strip()) < 10:
            continue  # Skip very short sentences (fewer than 10 chars)
        try:
            lang, conf = fasttext_predict(model, sent)
            iso_lang = to_iso3(lang)

            predictions.append(iso_lang)
            confidences.append(conf)

        except Exception:
            continue

    if not predictions:
        return "unknown", 0.0, {}  # Failsafe
    
    counts = Counter(predictions)
    final_lang = counts.most_common(1)[0][0]  # Majority vote
    avg_conf = sum(confidences)/len(confidences)

    return final_lang, avg_conf, dict(counts)

def to_iso3(lang_code):
    '''
    Normalize Fasttext predicted language code to ISO 639-3.
    Falls back to original value if unknown.
    '''
    return fasttext_to_iso3.get(lang_code, lang_code)

def load_robots(netloc, scheme):
    '''
    Get robots.txt from target website
    '''
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

# <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
def is_allowed(url, user_agent="*"):
    '''
    Checks whether specific URL is allowed to crawl as per robots.txt
    '''
    parsed = urlparse(url)
    netloc = normalize_netloc(parsed.netloc)
    scheme = parsed.scheme  # Scheme = http/https

    if netloc not in robots_cache:
        robots_cache[netloc] = load_robots(netloc, scheme)
    rp = robots_cache.get(netloc)

    return True if rp is None else rp.can_fetch(user_agent, url)  # Disclaimer: if robots.txt is not available, site is still crawled -- change to False to change behaviour

def make_unique_id(category, url):
    '''
    Create unique ID as metadata for each scraped text
    '''
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    raw = f"{category}|{base}".lower().strip()

    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def clean_text(soup):
    '''
    Cleans texts from noisy/redundant data.
    '''
    for tag in soup(["nav", "header", "footer", "script", "style", "img"]):  # Can be extended with other HTML tags if needed
        tag.decompose()

    for a in soup.find_all("a", href=True):

        href = a.get("href")

        if not isinstance(href, str):  # Ensure that href is a string
            continue

        href = href.strip()

        if not href:  # Previously crashed if href was None, this should fix it
            continue

        if href.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3")):
            a.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return " ".join(text.split())

def extract_date_from_url(url):
    '''
    Extract date from URL patterns like:
    /YYYY/MM/DD/
    /YYYY-MM-DD/
    /YYYY/MM/
    Returns a date string or None
    '''

    patterns = [
        r"/(20\d{2})[/-](\d{2})[/-](\d{2})/",  # /2026/01/27/ or /2026-01-27/
        r"/(20\d{2})[/-](\d{2})/"  # /2026/01
    ]

    for pattern in patterns:
        m = re.search(pattern, url)
        if not m:
            continue

        try:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3)) if len (m.groups()) == 3 else 1

            dt = datetime(year, month, day, tzinfo=timezone.utc)
            return dt.date().isoformat()
        
        except Exception:
            continue
    
    return None

def extract_publish_date(soup, url):
    '''
    Attempts to extract publish date and its source
    '''

    # Common meta tags for publish date extraction from HTML, possible to add new ones in the list if necessary
    meta_properties = [
        ("property", "article:published_time"),
        ("property", "article:published"),
        ("property", "og:published_time"),
        ("name", "datePublished"),
        ("name", "pubdate"),
        ("name", "dc.date"),
        ("name", "DC.date"),
        ("name", "publish-date"),
        ("name", "publish_date"),
        ("name", "created"),
        ("name", "rek:pubdate")
    ]

    # Metadata-based publish date extraction
    for attribute, value in meta_properties:
        tag = soup.find("meta", attrs={attribute: value})
        if tag and tag.get("content"):
            try:
                dt = datetime.fromisoformat(tag["content"].replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat(), "metadata"
            except Exception:
                pass

    # Check if tag contains time formatted data
    time_tag = soup.find("time", datetime=True)
    if time_tag:
        try:
            dt = datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat(), "metadata"
        except Exception:
            pass

    # URL-based date extraction
    url_date = extract_date_from_url(url)
    if url_date:
        return url_date, "url_pattern"

    return None, None  # Failsafe (if no tags were found)

def load_seen_hashes(output_path):
    '''
    Load previously seen text hashes from existing output file
    '''
    if not os.path.exists(output_path):
        return set()
    
    seen = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text_uid" in obj:
                    seen.add(obj["text_uid"])
            except Exception:
                continue
    return seen

#############
# Page parser
#############

def parse_page(html, url):
    soup = BeautifulSoup(html, "html.parser")

    published, published_source = extract_publish_date(soup, url)  # Extract publish date 

    # Extracting lang code from HTML code
    html_tag = soup.find("html")
    lang_html = html_tag.get("lang", "").lower() if html_tag else ""

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = clean_text(soup)

    # Storing (internal) child links to add to queue later
    links = []
    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"]).split("#")[0]
        if urlparse(link).netloc in allowed_domains:
            links.append(link)
    
    # Empty/non-text or too short
    if len(text) < 10:
        return None, links

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Strip text from title if text starts with title
    if title and text.startswith(title):
        text = text[len(title):].lstrip(" --: ")

    length = len(text)

    try:
        # Language prediction on the document level (whole text)
        if LANG_PREDICTION_LEVEL == "doc":
            fast_lang, conf = fasttext_predict(FAST_MODEL_ALL, text)
            sentence_stats = None
        
        # Language prediction on the sentence level
        else: 
            fast_lang, conf, sentence_stats = predict_language_sentence_level(text=text, model=FAST_MODEL_ALL)

    # Failsafe
    except Exception:
        fast_lang, conf = "unknown", 0.0
        sentence_stats = None

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
        "count_met": None,
        "count_tet": None,
        "count_het": None,
        "count_haan": None,
        "count_jokka": None
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

    # Output with metadata; links are not written to the output for smaller file size, only stored in the back-end
    return {
        "page_uid": uid,  # unique identifier for page
        "text_uid": text_hash,  # unique identifier for content (text)
        "url": url,
        "category": category,
        "lang-url-tag": lang_html or "unknown",
        "length": length,
        "lang-fasttext-identified": to_iso3(fast_lang),
        "lang-fasttext-confidence": conf,
        **rule_features,
        "lang_prediction_level": LANG_PREDICTION_LEVEL,
        "sentence_lang_distribution": sentence_stats,
        "final_prediction": final_prediction,
        "classification_type": classification_type,
        "crawl_timestamp": timestamp,
        "published": published,
        "published_source": published_source,
        "title": title,
        "text": text
    }, links

#############
# Worker (threaded)
#############

def crawl_one(url, outfile, seed_set):
    global seeds_processed

    netloc_norm = normalize_netloc(urlparse(url).netloc)
    crawl_stats["discovered"].add(url)

    if not is_allowed(url, "Mozilla/5.0"):
        log(f"Skipping {url} due to robots.txt")
        return []
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            log(f"Failed to crawl {url}: HTTP {r.status_code}")
            crawl_stats["failed"][url] = f"HTTP {r.status_code}"
            return []
    except requests.exceptions.RequestException as e:
        failure_type = classify_failure(e)
        log(f"Failed to crawl {url}: {e} ({failure_type})")
        crawl_stats["failed"][url] = failure_type
        return []

    # Feedback
    log(f"Crawling {url}")
    crawl_stats["crawled"].add(url)

    result = parse_page(r.text, url)
    
    # If no text (len < 10 chars)
    if result is None:  # Nothing usable on website
        log(f"Skipping {url} as no proper text was identified")
        return []
    
    page_data, child_links = result

    if page_data is None:  # In case a child link can be visited later from a skipped page
        return child_links

    with state_lock:
        if page_data["text_uid"] in seen_text_hashes:
            log(f"Skipping duplicate content at {url}")
            return child_links
        seen_text_hashes.add(page_data["text_uid"])

    with file_lock:
        json.dump(page_data, outfile, ensure_ascii=False)
        outfile.write("\n")
        outfile.flush()

    if url in seed_set:
        with state_lock:
            seeds_done.add(url)
            if not seeds_processed and seeds_done == seed_set:
                seeds_processed = True
                log("*** All seed URLs processed, now crawling child links ***")

    time.sleep(0.2)

    return child_links

#############
# Main
#############

def main():
    global FAST_MODEL_ALL, FAST_MODEL_FINFIT, DISAMBIGUATION_TYPE, fasttext_to_iso3, THREADING_ENABLED, seeds_processed, LANG_PREDICTION_LEVEL, seen_text_hashes, visited
    
    seen_text_hashes = load_seen_hashes(OUTPUT_FILE)
    log(f"Loaded {len(seen_text_hashes)} previously seen texts")

    log("Starting crawler")

    parser = argparse.ArgumentParser(description="Web crawler")
    parser.add_argument("-l", "--lang_level", help="Level of language prediction: doc (document) or sent (sentence)", choices=["doc", "sent"], default="doc")
    parser.add_argument("-f", "--finfit_model", help="Trained model for Finnish-Meänkieli disambiguation; Not required in case disambiguation-type = rule")
    parser.add_argument("-i", "--input", required=True, help="Input file containing target URLs")
    parser.add_argument(
        "-d", "--disambiguation_type",
        required=True,
        choices=["rule", "model"],
        help="Disambiguation type (rule or model) for Finnish/Meänkieli")
    
    # Threading
    parser.add_argument("-t", "--threading", action="store_true", help="Enable threaded crawling")
    parser.add_argument("-w", "--max_workers", type=int, default=4, help="Number of workers for threaded crawling (default=4)")

    args = parser.parse_args()

    THREADING_ENABLED = args.threading  # True or false

    # User feedback for missing argument
    if args.disambiguation_type == "model" and not args.finfit_model:
        parser.error("--finfit_model is required when --disambiguation-type=model")

    DISAMBIGUATION_TYPE = args.disambiguation_type
    LANG_PREDICTION_LEVEL = args.lang_level
    FAST_MODEL_ALL = fasttext.load_model(os.path.join(BASE_DIR, "models", "lid.176.ftz"))
    FAST_MODEL_FINFIT = None
    if DISAMBIGUATION_TYPE == "model":
        FAST_MODEL_FINFIT = fasttext.load_model(args.finfit_model)

    log("Fasttext model(s) loaded")

    with open(LANGCODES_PATH, "r", encoding="utf-8") as f:
        langcodes = json.load(f)

    fasttext_to_iso3 = {
        item["code-fasttext"]: item["code-iso"]
        for item in langcodes
    }

    # Opening and starting to process input file
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for category, block in data.items():
        urlcat = block.get("domain", category)  # Get metadata "Category"
        for entry in block.get("targets", []):
            url = entry.get("url")
            if not url:
                continue
            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url
            start_urls.append(url)
            urlcat_lookup[url] = urlcat
            allowed_domains.add(urlparse(url).netloc)

    # Previous solution
    start_urls[:] = list(set(start_urls))
    log(f"Loaded {len(start_urls)} seed URLs")

    seed_set = set(start_urls)
    seeds_processed = False

    visited = set()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:  # -a flag for appending to output

        # Without threading
        if not THREADING_ENABLED:
            log(f"Crawler running with no threading")

            queue = deque(start_urls)

            # Assign URL from queue
            while queue:
                url = queue.popleft()  
                if url in visited:
                    continue

                visited.add(url)

                child_links = crawl_one(url, outfile, seed_set)

                for link in child_links:
                    if link not in visited:
                        urlcat_lookup[link] = urlcat_lookup.get(url)  # For metadata
                        queue.append(link)

        # With threading
        else:
            log(f"Threading enabled with {args.max_workers} workers")
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:

                # Submitting initial URLs
                futures = {executor.submit(crawl_one, url, outfile, seed_set): url for url in start_urls}

                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)  # Wait for threads to complete
                    for future in done:
                        parent_url = futures.pop(future)
                        child_links = future.result()

                        # Submitting child links for crawling
                        for link in child_links:
                            if link not in visited:
                                visited.add(link)
                                urlcat_lookup[link] = urlcat_lookup.get(parent_url)  # Ensuring that the "category" field in the metadata gets populated correctly
                                futures[executor.submit(crawl_one, link, outfile, seed_set)] = link

    log("Crawl finished")
    log_file.close()

# Run only if run as a main program
if __name__ == "__main__":
    main()