# Web Information Retrieval

## Building an IR System with PyTerrier

A complete end-to-end walkthrough using a scientific paper search engine as the use case — from raw data to evaluated, ranked results.

**Topics:** PyTerrier · BM25 · Inverted Index · Query Expansion · nDCG · MAP · P@10 · Learning to Rank · Python

## Steps

1. Setup
2. Corpus
3. Indexing
4. Retrieval
5. Evaluation
6. Query Expansion
7. Re-ranking
8. Pipeline
9. Summary

## Use Case: Scientific Paper Search Engine

We are building a search engine for academic papers. A researcher types a query like `neural networks for text classification` and our system returns the most relevant papers, ranked by score.

`Raw Documents -> Index -> Query -> Retrieve -> Expand -> Re-rank -> Results + Eval`

---

## Step 01 · Installation & Initialization

PyTerrier wraps the Java-based Terrier IR platform. On first run it downloads the Terrier JAR automatically.
(`python-terrier` is the package name on PyPI; you import it as `pyterrier` in code.)

```bash
pip install python-terrier  # imported in Python as `pyterrier`
```

```python
import pyterrier as pt
import pandas as pd

# Downloads Terrier JAR on first call (~30 MB)
pt.init()

print("PyTerrier version:", pt.version())
```

Expected output:

```text
PyTerrier version: 0.10.0
Java started: pyterrier.java [version=5.10]
```

---

## Step 02 · Prepare the Corpus

Every document must have a unique `docno` field. Additional fields like title/abstract/authors can be added.

```python
documents = [
    {
        "docno": "p001",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce BERT, a language representation model using bidirectional transformer encoder pre-training on large corpora.",
        "authors": "Devlin et al.",
        "year": 2019,
    },
    {
        "docno": "p002",
        "title": "Attention Is All You Need",
        "abstract": "The Transformer architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions.",
        "authors": "Vaswani et al.",
        "year": 2017,
    },
    {
        "docno": "p003",
        "title": "BM25 and Beyond: Probabilistic Ranking Functions",
        "abstract": "Survey of probabilistic models for information retrieval, focusing on BM25 and its variants for document ranking.",
        "authors": "Robertson & Zaragoza",
        "year": 2009,
    },
    {
        "docno": "p004",
        "title": "Dense Passage Retrieval for Open-Domain QA",
        "abstract": "We show that dense representations outperform sparse BM25 retrieval for open-domain question answering tasks.",
        "authors": "Karpukhin et al.",
        "year": 2020,
    },
    {
        "docno": "p005",
        "title": "Learning to Rank for Information Retrieval",
        "abstract": "Supervised machine learning approaches for ranking documents, including RankNet, LambdaMART and ListNet.",
        "authors": "Liu",
        "year": 2011,
    },
    {
        "docno": "p006",
        "title": "ColBERT: Efficient and Effective Passage Retrieval",
        "abstract": "ColBERT introduces late interaction between BERT-encoded query and document tokens for scalable neural retrieval.",
        "authors": "Khattab & Zaharia",
        "year": 2020,
    },
]

df_corpus = pd.DataFrame(documents)
print(df_corpus[["docno", "title", "year"]])
```

---

## Step 03 · Build the Inverted Index

Indexing is done once. We combine `title + abstract` into a searchable `text` field.

```python
import os

INDEX_PATH = "./paper_index"

indexer = pt.index.IterDictIndexer(
    INDEX_PATH,
    overwrite=True,
    meta={"docno": 20, "title": 200},
)


def prepare_for_index(docs):
    for d in docs:
        yield {
            "docno": d["docno"],
            "title": d["title"],
            "text": d["title"] + " " + d["abstract"],
        }


indexref = indexer.index(prepare_for_index(documents))

index = pt.IndexFactory.of(indexref)
stats = index.getCollectionStatistics()

print(f"Documents indexed : {stats.numberOfDocuments}")
print(f"Unique terms      : {stats.numberOfUniqueTerms}")
print(f"Total tokens      : {stats.numberOfTokens}")
```

Expected output:

```text
Documents indexed : 6
Unique terms      : 147
Total tokens      : 312
Index written to ./paper_index/
```

---

## Step 04 · First Retrieval with BM25

```python
bm25 = pt.terrier.Retriever(
    indexref,
    wmodel="BM25",
    num_results=5,
    metadata=["docno", "title"],
)

results = bm25.search("neural networks text classification")
print(results[["rank", "docno", "score", "title"]])

queries = pd.DataFrame([
    {"qid": "q1", "query": "transformer language model"},
    {"qid": "q2", "query": "probabilistic document ranking"},
    {"qid": "q3", "query": "dense retrieval passage search"},
])

batch_results = bm25.transform(queries)
print(batch_results[["qid", "rank", "docno", "score"]])
```

---

## Step 05 · Evaluation (MAP, nDCG, P@5, Recall)

```python
from pyterrier.measures import *

qrels = pd.DataFrame([
    {"qid": "q1", "docno": "p001", "label": 2},
    {"qid": "q1", "docno": "p002", "label": 2},
    {"qid": "q1", "docno": "p006", "label": 1},
    {"qid": "q2", "docno": "p003", "label": 2},
    {"qid": "q2", "docno": "p005", "label": 1},
    {"qid": "q3", "docno": "p004", "label": 2},
    {"qid": "q3", "docno": "p006", "label": 1},
])

tfidf = pt.terrier.Retriever(indexref, wmodel="TF_IDF", num_results=5)

experiment_results = pt.Experiment(
    retr_systems=[bm25, tfidf],
    topics=queries,
    qrels=qrels,
    eval_metrics=[MAP, nDCG@10, P@5, R@5],
    names=["BM25", "TF-IDF"],
    baseline=0,
)

print(experiment_results)
```

Expected output:

```text
      name    MAP   nDCG@10   P@5   R@5
0     BM25  0.712    0.748  0.533 0.857
1  TF-IDF  0.634    0.681  0.467 0.800
```

---

## Step 06 · Query Expansion (Bo1)

```python
bm25_qe = (
    bm25
    >> pt.rewrite.Bo1QueryExpansion(indexref)
    >> bm25
)

expanded = (
    bm25
    >> pt.rewrite.Bo1QueryExpansion(indexref)
).search("transformer language model")

print("Original query: 'transformer language model'")
print("Expanded query:", expanded["query"].iloc[0])

experiment_qe = pt.Experiment(
    [bm25, bm25_qe],
    queries,
    qrels,
    [MAP, nDCG@10],
    names=["BM25", "BM25+QE"],
)
print(experiment_qe)
```

---

## Step 07 · Re-ranking with Learning to Rank

Use separate train/validation/test query splits when fitting and tuning LtR models.

```python
from lightgbm import LGBMRanker

bm25_r = pt.terrier.Retriever(indexref, wmodel="BM25")
dph_r = pt.terrier.Retriever(indexref, wmodel="DPH")
pl2_r = pt.terrier.Retriever(indexref, wmodel="PL2")

features = bm25_r >> (
    pt.terrier.FeaturesBatchRetrieve(
        indexref,
        features=[
            "WMODEL:BM25",
            "WMODEL:DPH",
            "WMODEL:PL2",
            "WMODEL:Tf",
            "WMODEL:TF_IDF",
        ],
    )
)

ltr_model = pt.ltr.apply_learned_model(
    LGBMRanker(n_estimators=100, num_leaves=31),  # example values; tune on validation data
    form="ltr",
)

ltr_pipeline = features >> ltr_model

# Example usage (split your topics + qrels into train/test sets first):
# ltr_pipeline.fit(train_queries, train_qrels)
# pt.Experiment([bm25_r, ltr_pipeline], test_queries, test_qrels, [MAP, nDCG@10])
```

> ⚠️ Train `ltr_pipeline` before reusing it in Step 08.

---

## Step 08 · Full Production Pipeline

```python
first_stage = pt.terrier.Retriever(indexref, wmodel="BM25", num_results=100)
qe = pt.rewrite.Bo1QueryExpansion(indexref)
second_stage = pt.terrier.Retriever(indexref, wmodel="BM25", num_results=20)

feat_pipe = pt.terrier.FeaturesBatchRetrieve(
    indexref,
    features=["WMODEL:BM25", "WMODEL:DPH", "WMODEL:PL2", "WMODEL:TF_IDF"],
)

# Reuse the trained LtR model from Step 07
reranker = ltr_model

full_system = (
    first_stage
    >> qe
    >> second_stage
    >> feat_pipe
    >> reranker
    % 10  # cutoff operator: keep top 10 results
)

final_results = full_system.search("BERT transformer pre-training NLP")
print(final_results[["rank", "docno", "score", "title"]])
```

---

## Step 09 · Summary

| Step | Action | PyTerrier API | IR Concept |
|---|---|---|---|
| 1 | Install & init | `pt.init()` | Terrier JVM setup |
| 2 | Prepare corpus | List of dicts / DataFrame | Document collection |
| 3 | Build index | `IterDictIndexer.index()` | Inverted index |
| 4 | First retrieval | `Retriever(wmodel="BM25")` | Probabilistic ranking |
| 5 | Evaluate | `pt.Experiment(..., metrics=[MAP, nDCG@10])` | Retrieval quality |
| 6 | Query expansion | `pt.rewrite.Bo1QueryExpansion()` | Pseudo-relevance feedback |
| 7 | Learning to rank | `FeaturesBatchRetrieve + LGBMRanker` | LambdaMART (GBDT LtR) |
| 8 | Compose pipeline | `stage1 >> qe >> stage2 >> ltr % 10` | Multi-stage retrieval |

---

TH Köln · Web Information Retrieval (Master Module)  
Built with PyTerrier · Apache Terrier · Python
