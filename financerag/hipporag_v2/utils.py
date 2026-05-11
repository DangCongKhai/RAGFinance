from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import logging

load_dotenv("../../.env")

GG_API_KEY = os.getenv("GOOGLE_API_KEY")


def create_vector_store():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", request_options={"timeout": 100000}
    )
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename="process.log")
formatter = logging.Formatter(
    "%(asctime)s :%(message)s", datefmt="%d/%m/%Y %I:%M:%S %p"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
