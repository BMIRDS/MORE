# Multi-Ontology Refined Embeddings (MORE): A Hybrid Multi-Ontology and Corpus-based Semantic Representation for Biomedical Concepts

By Steven Jiang and Saeed Hassanpour


![](./figures/MORE.png)

## Dependencies

# Usage

## 1. Corpora
### RadCore
RadCore is a large multi-institutional radiology report corpus for NLP. RadCore contains 1,899,482 reports, collected between 1995 and 2006, from three major healthcare organizations: Mayo Clinic (812 reports), MD Anderson Cancer Center (5000 reports), and Medical College of Wisconsin (1,893,670 reports).

### MIMIC-III
Medical Information Mart for Intensive Care (MIMIC-III) is a database containing information gathered from patients that were admitted to critical care units at a large hospital. MIMIC-III also contains a gold standard corpus of 2,434 ICU nursing notes accessible [here](https://mimic.physionet.org/gettingstarted/access/).

```
MIMIC-III, a freely accessible critical care database. Johnson AEW, 
Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, 
Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. 
Available from: http://www.nature.com/articles/sdata201635
```


## 2. UMLS
UMLS-Interface is a Perl package that provides an API to a local installation of the UMLS in a MySQL database (available [here](https://metacpan.org/pod/UMLS::Interface)). UMLS-Similarity is a Perl package that provides an API and a command line program to obtain the semantic similarity between CUIs in the UMLS given a specified set of source(s) and relations (available [here](https://metacpan.org/pod/UMLS::Similarity)). UMLS-Similarity contains five semantic similarity measures proposed by Rada, et. al., Wu & Palmer, Leacock & Chodorow, and Nguyen & Al-Mubaid, and the Path measure.

Installation instructions available [here](http://www.d.umn.edu/~tpederse/umls-similarity.html).

```
McInnes, B. T., Pedersen, T., & Pakhomov, S. V. (2009). 
UMLS-Interface and UMLS-Similarity: open source software for measuring paths and semantic similarity. 
In AMIA Annual Symposium Proceedings (Vol. 2009, p. 431). 
American Medical Informatics Association.
```

## 3. word2vec skip-gram
This work relies on the official Tensorflow implementation of the word2vec skip-gram model (available [here](https://github.com/tensorflow/models/tree/master/tutorials/embedding)).

## 4. Training

## 5. Evaluation



# Results




Physician Similarity Correlation             |  Expert Similarity Correlation
:-------------------------:|:-------------------------:
![](./figures/PhysicianGraph.png)  |  ![](./figures/ExpertGraph.png)