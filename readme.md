#  Readme

This is a README file with instructions on how to use the crawler. Table of contents:
#TODO

## Features

- Crawl a fixed set of target websites stored in an input file (JSON) #TODO: add required structure of the file?
- Respect access rules defined in `robots.txt` of each website
- Language detection of scraped texts with:
    1. an off-the-shelf Fasttext model with support for 176 languages
    2. additional Finnish/Meänkieli disambiguation with
        - a rule-based method, or
        - a trained Fasttext model specifically for Finnish/Meänkieli disambiguation
- Collecting metadata for each processed text
- Optional threaded crawling for more efficient processing
- Logging to console and a log file

## Usage

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
- `-d`, `--disambiguation-type`: strategy for Finnish/Meänkieli classification; accepted values are `model` or `rule`; if `model` is selected, then `-f` below needs to be defined
- `-f`, `--finfit-model`: location of the model used for Finnish/Meänkieli disambiguation
- `-t`, `--threading`: more efficient scraping with threading implemented
- `-w`, `--max_workers`: number of workers for threaded crawling (default=4)
