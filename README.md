# FISinFIS_Apriori_Python
Python implementation of FISinFIS Apriori algorithm for PAR and NAR generation from Frequent and Infrequent ItemSets as proposed by Mahmood et al. in their work *Negative and Positive Association Rules Mining from Text Using Frequent and Infrequent Itemsets* [[1]](#1).

## Requirements
- Python3
- [nltk](https://www.nltk.org/)
- [networkx](https://networkx.org/)
- pandas
- seaborn

## Usage
The following examples require to change directory to FISinFIS_Apriori_Python/code before running the code.
```bash
cd FISinFIS_Apriori_Python/code
```
### Command Line
```bash
python3 fisinfis.py -topN TOPN -ms MINSUPP -mc MINCONF -MIDF MAXIDF_PERC -ml MINLIFT -df "DATASET_FOLDER_PATH" -csv "DATASET.CSV" -vc "VERBOSE_COLUMN1" "VERBOSE_COLUMNS2" -lang "LANGUAGE"
```
example:
```bash
 python3 fisinfis.py -topN 0.99 -ms 0.05 -mc 0.6 -MIDF 95 -ml 1.01 -df "/Users/alessiomongelluzzo/Downloads/archive/" -csv "snli_1.0_train.csv" -vc sentence1 sentence2 -lang "English"
 ```
### Import
N.B.: the following example application was performed on the first 100 rows of Stanford Natural Language Inference Corpus train dataset publicly available [here](https://www.kaggle.com/stanfordu/stanford-natural-language-inference-corpus).
```python
from fisinfis import FISinFIS
f = FISinFIS(topN = 0.99, min_supp=0.05, min_conf=0.6, max_IDF_percentile=95, min_lift=1.01)
f.initialize(data_folder="/archive/", csv_name="snli_1.0_train.csv",
             verbose_cols=["sentence1", "sentence2"])
f.cleanse(lang="english")
```
![top30](https://github.com/AlessioMongelluzzo/FISinFIS_Apriori_Python/blob/master/examples/cleanse_top30.jpg)
```python
f.move_to_sparse()
f.algorithm1()
```
![algo11](https://github.com/AlessioMongelluzzo/FISinFIS_Apriori_Python/blob/master/examples/algo11_idf.jpg)
![algo12](https://github.com/AlessioMongelluzzo/FISinFIS_Apriori_Python/blob/master/examples/algo12_sup.jpg)
```python
f.algorithm2()
f.plot_ARs_graphs() # saves AR graph representation on disk
```
![plotar](https://github.com/AlessioMongelluzzo/FISinFIS_Apriori_Python/blob/master/examples/AR_game.jpg)
```python
f.show_stem_transactions(["walk", "pizza"]) # shows info about items and transactions containing stemmed items
```
![show_stem](https://github.com/AlessioMongelluzzo/FISinFIS_Apriori_Python/blob/master/examples/show_stem.jpg)
## Reference
<a id="1">[1]</a> 
Mahmood, S., Shahbaz, M., & Guergachi, A. (2014). Negative and positive association rules mining from text using frequent and infrequent itemsets. The Scientific World Journal, 2014.
