# Finance RAG Project

## 1. Set up:

Install dependency for projects:
```bash
pip install requirements.txt
```


Create an environment file `.env` and add variables as followed

```bash
GOOGLE_API_KEY = <your_google_api_key_here>
PINECONE_API_KEY = <your_pinecone_api_key_here>
```

## 2. Next step

1. Split corpus into chunks for faster loading, retrieve more top-k most relevant documents and merge those documents to form final answer!

2. Add BM25 retriever to create hybrid search

3. Use a cross-encoder at the end to rerank retrieved documents

