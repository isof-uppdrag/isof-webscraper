#  Readme

This is a README file with instructions on how to use the crawler. Table of contents:

### 1. [Features](#features)
### 2. [Getting started](#getting-started)
### 3. [Process flow](#process-flow)
### 4. [Output](#output)
### 5. [Additional resources](#additional-resources)
### 6. [Lexical features used for rule-based Meänkieli/Finnish disambiguation](#lexical-features-used-for-meänkielifinnish-rule-based-disambiguation)

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

4. The crawler is tracking the visited pages and the scraped content to handle duplicates. The output files are written incrementally and whenever you run the program, new data is appended to the files.

## Output

The scraped texts are stored in a JSON file in the `output` folder with the following metadata:

- `page_uid`: a (hashed) unique identifer for the page
- `text_uid`: a (hashed) unique identifier for the content (text)
- `url`: the URL from which the text was scraped
- `category`: type of website (e.g. kommun/region/myndighet/radio) -- in the current implementation the only website with the *radio* tag is *https://www.sverigesradio.se/meanraatio*
- `lang-url-tag`: the language tag from the website's HTML code
- `length`: length of the text in number of characters
- `lang-fasttext-identified`: ISO 639-3 code predicted with the off-the-shelf model (not necessarily the final language tag)
- `lang-fasttext-confidence`: prediction confidence of the off-the-shelf model [0-1]
- A set of frequencies of various features used for rule-based Finnish/Meänkieli disambiguation (e.g. relative frequency of letter *d* and *h*, counts of Meänkieli-specific lexical items etc.) NB: only gets populated when `--disambiguation_type` is set to `rule`, otherwise `null` is written to all features. For more information about rules and for changing these, see `langclassifier.py` - the script from which the rule-based function is imported in the crawler. For more information about these features, see chapter **[Lexical features used for Meänkieli/Finnish rule-based disambiguation](#lexical-features-used-for-rule-based-meänkielifinnish-disambiguation)**
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

## Additional resources

In the folder *additional resources*, you can find a json file including all the unique `lang` tags that were found in the HTML code of the target websites. This can be useful for further langtech tasks.

## Lexical features used for rule-based Meänkieli/Finnish disambiguation

When the `--disambiguation-type` flag is set to `rule`, the crawler is using a function imported from `langclassifier.py` to classify Finnish and Meänkieli texts using a rule based on a set of differentiating features.

Features not currently used by the classifier but could be interesting for future use cases:
- `count_d`: the number of times the letter *d* occurs in the text
- `rel_freq_d`: relative frequency of the letter *d* in the text
- `count_h`: the number of times the letter *h* occurs in the text
- `rel_freq_h`: relative frequency of the letter *hÄ in the text

Features currently used by the classifier:
- `count_ette`: the number of times the lexical item *ette* occurs in the text
- `count_oon`: the number of times the lexical item *oon* occurs in the text
- `count_mie`: the number of times the lexical item *mie* occurs in the text
- `count_sie`: the number of times the lexical item *sie* occurs in the text
- `count_met`: the number of times the lexical item *met* occurs in the text
- `count_tet`: the number of times the lexical item *tet* occurs in the text
- `count_het`: the number of times the lexical item *het* occurs in the text
- `count_haan`: the number of times the lexical item *hään* occurs in the text
- `count_jokka`: the number of times the lexical item *jokka* occurs in the text

Then, the tag `fit` (Meänkieli) tag will be assigned as `final_prediction` when the below conditions are met:

```
if any(x > 0 for x in (count_ette, count_oon, count_mie, count_sie, count_met, count_tet, count_het, count_haan, count_jokka))
````

In a test of 89 Finnish and 67 Meänkieli texts, this rule gave an accuracy of 99,4 % but the rule can easily be changed in `langclassifier.py` if necessary.