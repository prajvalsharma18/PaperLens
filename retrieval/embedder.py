"""
Embedding model wrapper.

Swap out the model by changing EMBED_MODEL in .env.
"""

import os
from langchain_openai import OpenAIEmbeddings


def get_embedder() -> OpenAIEmbeddings:
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)
