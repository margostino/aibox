from pathlib import Path
from llama_index import download_loader
from llama_index import VectorStoreIndex
from dotenv import load_dotenv

load_dotenv()

SimpleCSVReader = download_loader("SimpleCSVReader")

file_path = "/Users/margostino/workspace/aibox/data/cereal.csv"
loader = SimpleCSVReader(encoding="utf-8")
documents = loader.load_data(file=Path(file_path))

# index = VectorStoreIndex.from_documents(documents, show_progress=True)
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What is the Cereal yield index (World Bank (2017) & OWID) in Afghanistan in 1962")
print(response)