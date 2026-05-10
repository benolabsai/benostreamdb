import json

with open('demo/02_rag_pipeline.ipynb', 'r') as f:
    nb = json.load(f)

# The cell at index 4 (5th cell) is the ingestion cell
ingest_cell = nb['cells'][4]
ingest_cell['source'] = [
    "dataset = load_dataset(\"squad\", split=\"train\")\n",
    "unique_contexts = pd.DataFrame(dataset)[\"context\"].unique()[:300]\n",
    "df = pd.DataFrame({\"context\": unique_contexts})\n",
    "df[\"title\"] = [dataset[i][\"title\"] for i in range(300)]  # Roughly map some titles\n",
    "\n",
    "print(f\"Embedding {len(df)} Wikipedia paragraphs...\")\n",
    "# Use numpy arrays directly - HyperStreamDB accepts them natively\n",
    "embeddings = model.encode(df[\"context\"].tolist())\n",
    "print(f\"\u2705 Embeddings format: {type(embeddings[0])}, Shape: {embeddings[0].shape}, Dtype: {embeddings[0].dtype}\")\n",
    "df[\"embedding\"] = list(embeddings)  # Each row gets a numpy array directly\n",
    "\n",
    "# Write to HyperStreamDB\n",
    "\n",
    "table = hdb.Table(\"./rag_db\")\n",
    "table.add_index_columns([\"embedding\"])\n",
    "table.write_pandas(df)\n",
    "table.commit() # Create partition layout out of index definitions!\n",
    "print(\"Knowledge base ingested with natively built semantic graph.\")"
]

with open('demo/02_rag_pipeline.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
