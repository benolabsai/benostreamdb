import sys
import numpy as np
sys.path.insert(0, './python')
from hyperstreamdb import Table
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
table = Table("./news_db")
query = "Winning medals in international sports competitions"
query_embedding = list(model.encode(query))

hybrid_results = table.filter(
    filter="category = 'Sci/Tech'",
    vector_filter=query_embedding,
    k=5).to_pandas()
print(hybrid_results[["category", "text", "distance"]])
