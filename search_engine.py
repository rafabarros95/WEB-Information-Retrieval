import os
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
import ir_datasets
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

pt.java.init()

print("+"*80)
# STEP 1 - Load dataset
print("+"*80)

dataset = pt.get_dataset("irds:cord19/trec-covid")

# Load topics manually via ir_datasets (bypasses PyTerrier/pandas bug)
irds = ir_datasets.load("cord19/trec-covid")

topics = pd.DataFrame([
    {"qid": str(q.query_id), "query": q.title}
    for q in irds.queries_iter()
])

qrels = dataset.get_qrels()

print(topics.head())
print(qrels.head())
print(f"Number of queries : {len(topics)}")
print(f"Judged pairs      : {len(qrels)}")
print(f"Papers            : 192509")

print("+"*80)
# STEP 2 - Build the index
print("+"*80)

INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cord19_index")

# Create the directory if it doesn't exist
os.makedirs(INDEX_PATH, exist_ok=True)

if not os.path.exists(INDEX_PATH + "/data.properties"):
    print("Building index... (first time only, ~5-10 mins)")
    indexer = pt.IterDictIndexer(
        INDEX_PATH,
        overwrite=True,
        meta={"docno": 26},
    )

    def corpus_iter():
        for doc in dataset.get_corpus_iter():
            yield {
                "docno": doc["docno"],
                # Combine title + abstract into a single "text" field
                "text": (doc.get("title", "") or "") + " " + (doc.get("abstract", "") or "")
            }

    indexref = indexer.index(corpus_iter())
    print("✓ Index built!")
else:
    print("✓ Index already exists, loading from disk...")
    indexref = pt.IndexRef.of(INDEX_PATH + "/data.properties")

# Inspect the index
index = pt.IndexFactory.of(indexref)
stats = index.getCollectionStatistics()
print(f"\nIndex stats:")
print(f"  Documents   : {stats.numberOfDocuments}")
print(f"  Unique terms: {stats.numberOfUniqueTerms}")
print(f"  Total tokens: {stats.numberOfTokens}")

print("+"*80)
# STEP 3 - BM25 retrieval
print("+"*80)

bm25 = pt.terrier.Retriever(
    indexref,
    wmodel="BM25",
    num_results=1000,
    metadata=["docno"]
)

print("\nTest search: 'coronavirus origin'")
print(bm25.search("coronavirus origin").head(5))

print("+"*80)
# STEP 4 - TF-IDF
print("+"*80)

tfidf = pt.terrier.Retriever(
    indexref,
    wmodel="TF_IDF",
    num_results=1000,
    metadata=["docno"]
)

print("+"*80)
# STEP 5 - BM25 + Query Expansion
print("+"*80)

bm25_qe = (
    bm25
    >> pt.rewrite.Bo1QueryExpansion(indexref)
    >> bm25
)

print("+"*80)
# STEP 6 - Evaluate all 3 systems
print("+"*80)

print("\nRunning experiment...")

results = pt.Experiment(
    [bm25, tfidf, bm25_qe],
    topics,
    qrels,
    eval_metrics=[MAP, nDCG@10, P@10, R@1000],
    names=["BM25", "TF-IDF", "BM25+QE"],
    baseline=0,
    correction="bonferroni"
)

print("\n=== Experiment Results ===")
print(results.to_string(index=False))