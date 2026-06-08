# synthetic_translation_bbc
# Batak Wikipedia Synthetic Translation Pipeline

This project extracts clean Batak Toba (`bbc`) sentences from a cleaned Wikipedia JSONL file, translates them using `googletrans`, and optionally reverses the translation direction for training an `id/en → bbc` model.

## 1. Install dependencies

Create and activate your environment first, then install:

```bash
conda create -n batak-mt python=3.11 -y
conda activate batak-mt
pip install googletrans tqdm ftfy
```


## 2. Running Batak Toba <-> Indonesian/English translation generation

Arguments --total for how many translations
Arguments --id-ratio for ratio. Default is id:70% and en:30%

```bash
python generate_googletrans_wiki.py --total 50000
```

## 3. Running Indonesian/English <-> Batak Toba translation generation

Arguments --total for how many translations.
Arguments --id-ratio for ratio. Default is id:70% and en:30%
Arguments --start-row for starting row of input sentence rows. This value should indicate success + fail of previous run, to ensure no duplicate input

```bash
python generate_googletrans.py --total 50000 --start-row 50197
```
