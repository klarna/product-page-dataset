# The Klarna Product Page Dataset
## Description
The Klarna Product Page Dataset is a dataset of publicly available pages corresponding to products sold online on
various e-commerce websites. The dataset contains offline snapshots of 51,701 product pages collected from
8,175 distinct merchants across 8 different markets (US, GB, SE, NL, FI, NO, DE, AT) between 2018 and 2019.
On each page, analysts labelled 5 elements of interest: the price of the product, its image, its name and the add-to-cart
and go-to-cart buttons (if found). These labels are present in the HTML code as an attribute called `klarna-ai-label` taking one of the values: `Price`, `Name`, `Main picture`, `Add to cart` and `Cart`.

The snapshots are available in 2 formats: as MHTML files and as [WebTraversalLibrary](https://github.com/klarna-incubator/webtraversallibrary) (WTL) snapshots.
The MHTML format is less lossy, a browser can render these pages though any Javascript on the page is lost.
The WTL snapshots are produced by loading the MHTML pages into a chromium-based browser. To keep the WTL dataset more compact, they do not contain the screenshot of the rendered MTHML, but only its HTML, and page and element metadata with additional rendering information (bounding boxes of elements, font sizes etc.).
For convenience, the datasets are provided with a train/test split in which no merchants in the test set are present in the training set.

## Corresponding Publication
For more information about the contents of the datasets (statistics etc.) please refer to the following [ArXive paper](https://arxiv.org/abs/2111.02168).


## Download under the [Creative Commons BY-NC-SA license](https://creativecommons.org/licenses/by-nc-sa/4.0/):
### WTL format (~8GB):
https://klarna-research-public-datasets.s3-eu-west-1.amazonaws.com/klarna_product_page_dataset_50k/klarna_product_page_dataset_WTL_50k.tar.gz

### MHTML format (~51GB):
https://klarna-research-public-datasets.s3-eu-west-1.amazonaws.com/klarna_product_page_dataset_50k/klarna_product_page_dataset_MHTML_50k.tar.gz

### You can also find individual tarballs per market (in both formats) in this bucket (on region `eu-west-1`). Structure of the bucket is as follows:
```
s3://klarna-research-public-datasets 
└── klarna_product_page_dataset_50k
    ├── klarna_product_page_dataset_MHTML_50k
    │   ├── AT_test.tar.gz
    │   ├── AT_train.tar.gz
    │   ├── DE_test.tar.gz
    │   ├── DE_train.tar.gz
    │   ├── FI_test.tar.gz
    │   ├── FI_train.tar.gz
    │   ├── GB_test.tar.gz
    │   ├── GB_train.tar.gz
    │   ├── NL_test.tar.gz
    │   ├── NL_train.tar.gz
    │   ├── NO_test.tar.gz
    │   ├── NO_train.tar.gz
    │   ├── SE_test.tar.gz
    │   ├── SE_train.tar.gz
    │   ├── US_test.tar.gz
    │   └── US_train.tar.gz
    ├── klarna_product_page_dataset_MHTML_50k.tar.gz
    ├── klarna_product_page_dataset_WTL_50k
    │   ├── AT_test.tar.gz
    │   ├── AT_train.tar.gz
    │   ├── DE_test.tar.gz
    │   ├── DE_train.tar.gz
    │   ├── FI_test.tar.gz
    │   ├── FI_train.tar.gz
    │   ├── GB_test.tar.gz
    │   ├── GB_train.tar.gz
    │   ├── NL_test.tar.gz
    │   ├── NL_train.tar.gz
    │   ├── NO_test.tar.gz
    │   ├── NO_train.tar.gz
    │   ├── SE_test.tar.gz
    │   ├── SE_train.tar.gz
    │   ├── US_test.tar.gz
    │   └── US_train.tar.gz
    └── klarna_product_page_dataset_WTL_50k.tar.gz
```
# Datasheets for Datasets Documentation

## Motivation

### For what purpose was the dataset created?
The dataset was created for the purpose of benchmarking representation learning algorithms on the task of web element prediction on e-commerce websites.
Specifically, the dataset provides algorithms with a large-scale, diverse and realistic corpus of labelled product pages containing both DOM-tree representations but also page screenshots.
Previous datasets used in evaluations of DOM-element prediction algorithms contained pages generated from very few (only 80) templates.
The Klarna Product Page Dataset contains a 51,701 product pages belonging to 8,175 websites, allowing algorithms to potentially learn more complex and abstract representations on one vertical - e-commerce product pages.

### Who created the dataset and on behalf of which entity?
The dataset is created by the Web Automation Research team within Klarna.

## Composition

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
The dataset contains offline copies of publicly available e-commerce product web pages.

### How many instances are there in total (of each type, if appropriate)?
The dataset contains 51,701 product pages belonging to 8,175 websites over 8 different markets.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The dataset is a sample of all existing product pages available in 8 markets.
Due to limited resources, the dataset is not representative of the wider set of all possible product web pages you might encounter over the internet since:
* It contains only a fraction of all product pages available
* It contains only 8 markets, not equally represented
* It contains a bias towards popular/high-volume merchants
* It contains predominantly pages rendered in a mobile view

### What data does each instance consist of?
Each example consists of a MHTML or [WebTraversalLibrary](https://github.com/klarna-incubator/webtraversallibrary)
(WTL) clone of the loaded page.

### Is there a label or target associated with each instance?
On each page, there are 5 elements of interest: the price of the product, its image, its name and the add-to-cart
and go-to-cart buttons (if found).
These labels are present in the HTML code as an attribute called `klarna-ai-label` taking one of the values:
`Price`, `Name`, `Main picture`, `Add to cart` and `Cart`.

### Is any information missing from individual instances?
Some pages might be empty due to errors when loading.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?
Yes, pages belonging to the same market appear in the same directory. Pages belonging to the same merchant also appear in the same directory.

### Are there recommended data splits (e.g., training, development/validation, testing)?
Yes, the dataset is presented in a pre-made split train/test split (with ratios 0.8, 0.2).
The recommended split is done such that no merchant appears in more than 1 set so that no webpage template is present in 2 sets in order to gauge the algorithms' ability to generalise.

### Are there any errors, sources of noise, or redundancies in the dataset?
* Labels can be placed on elements that appear as transparent layers on top of the actual elements on interest.
For instance in the case of the product image, if they appear as part of a carousel, the label will appear on a transparent layer on top of the actual image.
* Pages contain a single label, despite there being more than 1 acceptable predictions for an element. For instance, a parent element of the price element could contain the same text, or a child of the `Add to cart` button could trigger the same event.
* Some pages might have not loaded properly or might appear empty.
* Javascript execution is lost when an MHTML clone is saved. Parts of the webpages that rely on Javascript execution would then not load properly.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?
Yes. The dataset is made available via the Registry of Open Data on AWS.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctorpatient confidentiality, data that includes the content of individuals’ non-public communications)?
No. Dataset contains only publicly available webpages.

## Collection Process

### How was the data associated with each instance acquired?
The data was directly observable.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
The sampling strategy was deterministic for merchants - the product pages were sampled curated by the analysts labelling the pages with a bias towards pages corresponding to product that did not require configuring (e.g. one-size fits all, default size selection etc.).

### Over what timeframe was the data collected?
The data instances in the dataset were collected between 2018 and 2019.

## Preprocessing

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?
In the case of the MHTML dataset, no. For the dataset in the WTL, the element metadata additionally contains the local node text, extracted from the source HTML.
