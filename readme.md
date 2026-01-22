#  Readme

This is a README file with instructions on how to use the crawler. Table of contents:

### 1. [Features](#features)
### 2. [Getting started](#getting-started)
### 3. [Process flow](#process-flow)
### 4. [Output](#output)

## Features

- Crawl a fixed set of target websites stored in an input file (JSON)
- Respect access rules defined in `robots.txt` of each website
- Language detection of scraped texts with:
    1. an off-the-shelf Fasttext model with support for 176 languages (source/more information: https://fasttext.cc/docs/en/language-identification.html)
    2. additional Finnish/Meänkieli disambiguation with
        - a rule-based method, or
        - a trained Fasttext model specifically for Finnish/Meänkieli disambiguation
- Collecting metadata for each processed text
- Optional threaded crawling for more efficient processing
- Logging to console and a log file

## Getting started

Install all required libraries by running
```
pip install -r requirements.txt
```

Create a `models` folder and download the off-the-shelf language classification model from Fasttext by running:
```
mkdir -p models && curl -L -o models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

If you are planning to use a classifier model for Finnish/Meänkieli disambiguation, you can train a Fasttext model by running:
```
python3 model.py
```

This will save a model called `fin-fit-disambiguation-model` in the `models` folder that you created in the previous step.

The crawler has a number of flags/command line arguments that define the functionality:
- `-i`, `--input`: location of the input file (JSON) containing the target websites
- `-l`, `--lang_level`: level at which language prediction should be done (choices:["doc", "sent"]) -- doc = whole document (text), sent = sentence
- `-d`, `--disambiguation_type`: strategy for Finnish/Meänkieli classification; accepted values are `model` or `rule`; if `model` is selected, then `-f` below needs to be defined
- `-f`, `--finfit_model`: location of the model used for Finnish/Meänkieli disambiguation
- `-t`, `--threading`: more efficient scraping with threading implemented
- `-w`, `--max_workers`: number of workers for threaded crawling (default=4)

    ### Example command lines

    **1. Threaded crawling (4 workers) with model disambiguation and document level language prediction**
    ```
    python3 isof-crawler.py -f models/fin-fit-disambiguation-model.bin -i json/targets.json -d model -t -w 4 -l doc
    ```

    **2. Sequential crawling with rule-based disambiguation and sentence level language prediction**
    ```
    python3 isof-crawler.py -i json/targets.json -d rule -l sent
    ```

When you start running the program, two files will be saved in the newly created `output` folder:
1. `scraped-corpus.json`: the file containing all the scraped texts with the corresponding metadata
2. `crawler.log`: log file storing all the user feedback during the crawling process with timestamps

## Process flow

1. The crawler starts going through the seed URL's from the input file and saves internal child links from each website.
2. Metadata and texts are written to the output JSON file. The language prediction is stored in the metadata variable `final_prediction`
3. The scraped texts also get passed through a pre-trained language identification model (Fasttext) for an initial language prediction. The model has support for 176 languages but does not include minority languages such as Meänkieli, Romani or Sami. To mitigate that, the following procedure is applied:
    - If the predicted language is Finnish (`fin`), the text is checked by either a rule-based algorithm (`-d rule`) or a model trained specifically for Finnish/Meänkieli disambiguation (`-d model`). Since Meänkieli texts are most likely to be predicted as Finnish by the first model, this is an extra step to ensure that we separate the actual Finnish/Meänkieli texts from each other.
    - In case the `lang` tag in the scraped website's HTML code is an instance of `{"rmf", "rmn", "rmy", "rmu", "rom"}`, then `final_prediction` is overwritten to the ISO 639-3 code `rom`.
    - In case the `lang` tag in the scraped website's HTML codde is an instance of `{"smi", "smj", "sma", "sme", "se"}` or if the URL in question includes the string `samegi` (short for *samegiella*), then `final_prediction` is overwritten to the ISO 639-3 code `smi`.

## Output

The scraped texts are stored in a JSON file in the `output` folder with the following metadata:

- `uid`: a (hashed) unique identifer for the text
- `url`: the URL from which the text was scraped
- `category`: type of website (e.g. kommun/region/myndighet/radio) -- in the current implementation the only website with the *radio* tag is *https://www.sverigesradio.se/meanraatio*
- `lang-url-tag`: the language tag from the website's HTML code
- `length`: length of the text in number of characters
- `lang-fasttext-identified`: ISO 639-3 code predicted with the off-the-shelf model (not necessarily the final language tag)
- `lang-fasttext-confidence`: prediction confidence of the off-the-shelf model [0-1]
- A set of frequencies of various features used for rule-based Finnish/Meänkieli disambiguation (e.g. relative frequency of letter *d* and *h*, counts of Meänkieli-specific lexical items etc.) NB: only gets populated when `--disambiguation_type` is set to `rule`, otherwise `null` is written to all features. For more information about rules and for changing these, see `langclassifier.py` - the script from which the rule-based function is imported in the crawler.
- `lang_prediction_level`: the level at which the language prediction was done (document/sentence)
- `sentence_lang_distribution`: the distribution of the identified languages on the sentence level for the whole document (text)
- `final_prediction`: the final language tag in ISO 639-3 format
- `classification_type`: information about how the final prediction was done (e.g. *Fasttext off-the-shelf/Rule-based/Fasttext trained disambiguator model/HTML overwrite*)
- `crawl_timestamp`: the timestamp at which the crawler exported the text
- `published`: the publication date of the text (*null* when not available)
- `title`: the title of the scraped text
- `text`: the actual text

    ### Parsing the output

    Using the file `output-parser.py`, you can parse the scraped corpus and create sub-corpora for various languages specifically.