from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint


llm = Ollama(model="llama2:7b")

# # res = llm("Tell me about the history of AI")
# # print(res)


# # Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('../../../state_of_the_union.txt').load()


file_path='/Users/margostino/workspace/aibox/data/fpl_bootstrap_static.json'
# data = json.loads(Path(file_path).read_text())
# pprint(data)

loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',
    text_content=False)

documents = loader.load()
# pprint(data)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OllamaEmbeddings())

query = "Give me information about a random player"
docs = db.similarity_search(query)
print(docs[0].page_content)