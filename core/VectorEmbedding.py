import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

class VectorEmbedding:
    def __init__(self):
        """Initializes the shared OpenAI client for vector transformation."""
        api_key=os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ Environment Error: Missing OPENAI_API_KEY value inside environment configurations.")

        self.client = OpenAI(api_key=api_key)

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Calls OpenAI API to generate 1536-dim vector arrays for a batch of text chunks."""
        if not texts:
            return []
        print(f"🧬 Generating Vector Embeddings for {len(texts)} text blocks...")

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [data.embedding for data in sorted(response.data, key=lambda x: x.index)]