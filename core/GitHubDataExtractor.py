import os
import sys
from pathlib import Path
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from commons.DataValidation import DynamicRepoMetadata, READMEChunk, RepoDocument
from commons.chunk import recursive_text_splitter
from core.GitHubConnector.GitHubAPIConnector import GitHubConnector
from core.VectorEmbedding import VectorEmbedding
from openai import OpenAI
from dotenv import load_dotenv
import httpx
import asyncio

load_dotenv(override=True)

class GitHubDataExtractor:
    def __init__(self):
        self.client = OpenAI()
        self.github_client = GitHubConnector()
        self.username = "ceewa30"
        self.embedding_client = VectorEmbedding()

    async def github_extractor(self):
        # 1. Fetch skills (this hits the API once)
        self.github_skills = await self.github_client.fetch_skills(username=self.username)

        # 2. Fetch repo names directly to trigger README downloads
        self.repo_names = await self.github_client.fetch_repo_names(username=self.username)

        # 3. Download all READMEs concurrently using a single client session
        async with httpx.AsyncClient() as client:
            tasks = [self.github_client.fetch_readme_raw(client, self.username, repo) for repo in self.repo_names]
            results = await asyncio.gather(*tasks)

            # Filter down into a dictionary containing your actual README content strings
            self.captured_readmes = {r["repo"]: r["readme"] for r in results if r["readme"]}

        print("\n========================================================")
        print("             GITHUB INGESTION RAW JSON DATA              ")
        print("========================================================")
        print("========================================================\n")

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        # Track the global collection of all structured document model structures
        all_repo_documents = []

        # Execute dynamic mapping engine loop
        for repo_name, readme_content in self.captured_readmes.items():
            # Step A: Parse the text automatically using the LLM extraction wrapper
            root_dir = Path(__file__).resolve().parent
            if (root_dir / "resources").exists():
                project_root = root_dir
            else:
                project_root = root_dir.parent

            prompt_path = project_root / "resources" / "system_prompt.txt"
            print(f"Loading prompt from: {prompt_path}")

            # Open and read the raw file content
            with open(prompt_path, "r", encoding="utf-8") as f:
                raw_content = f.read().strip()

            pattern = r'"document"\s*:\s*"(.*?)"\s*,\s*"github"'
            match = re.search(pattern, raw_content, re.DOTALL)

            if match:
                system_prompt = match.group(1).strip()
                # print("\n--- Extracted Document Prompt Safely ---")
                # print(system_prompt)
            else:
                print("Error: Could not locate the 'document' prompt structure in your file.")

            prompt = f"""
            Analyze the raw markdown content of the repository '{repo_name}' and extract structural metadata features.

            RAW README TEXT:
            {readme_content}
            """

            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=DynamicRepoMetadata
            )
            extracted_meta =  completion.choices[0].message.parsed

            # Step B: Format the schema directly into a flat Chroma metadata-compliant dictionary
            flat_metadata = {
                "repository_name": repo_name,
                "author": extracted_meta.author,
                "primary_stack": extracted_meta.tech_stack,
                "ml_used": extracted_meta.uses_machine_learning,
                "algo_type": extracted_meta.algorithm_summary,
                "current_role": extracted_meta.current_role,
                "file_type": "github"
            }

            print(f"\n🧩 Breaking down README text segments for repository: '{repo_name}'")
            # 🚀 Step C: Segment the large raw file content text down to 800-character fragments
            text_chunks = recursive_text_splitter(readme_content, chunk_size=800, chunk_overlap=150)
            print(f"  -> Generated {len(text_chunks)} chunk rows.")

            # 🚀 Step D: Convert these local text fragments into 1536-dim vector arrays in bulk
            chunk_vectors = self.embedding_client.generate_batch_embeddings(text_chunks)


            # Local collection for this specific repository document mapping layer
            current_repo_readme_chunks = []

            # Step E: Instantiate models and populate flat lists for database loading
            for idx, chunk_text in enumerate(text_chunks):
                # Form individual unique row identifiers

                row_id = f"repo_{repo_name.lower()}_chunk_{idx}".replace('/', '_').replace('-', '_')

                # Match the flat meta fields but keep positional indexes tracked cleanly
                chunk_specific_meta = flat_metadata.copy()
                chunk_specific_meta["chunk_index"] = idx



                # Instantiate Pydantic object to guarantee structural data integrity
                chunk_obj = READMEChunk(
                    chunk_id=row_id,
                    chunk_index=idx,                  # Native primitive int (Safe)
                    content=chunk_text,
                    metadata=chunk_specific_meta,
                    embedding=chunk_vectors[idx]      # Clean 1536-float row array assignment
                )
                current_repo_readme_chunks.append(chunk_obj)

                # Feed straight into flat arrays ready for ChromaDB consumption
                documents.append(chunk_text)
                metadatas.append(chunk_specific_meta)
                ids.append(row_id)
                embeddings.append(chunk_vectors[idx])

            # Step F: Wrap chunks inside your imported RepoDocument structural asset
            repo_doc_asset = RepoDocument(
                repo_name=f"{repo_name}_repository",
                raw_readme=readme_content,
                readme_chunks=current_repo_readme_chunks
            )
            all_repo_documents.append(repo_doc_asset)
            # print(f"✅ Successfully compiled {len(current_repo_readme_chunks)} rows into '{repo_doc_asset.repo_name}'")

        github_payload = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
            "embeddings": embeddings,
            "repo_documents": all_repo_documents
        }
        # print(f"Github Metadata: {github_payload["metadatas"]}")
        return github_payload

if __name__=="__main__":
    github_data_extractor = GitHubDataExtractor()

    github_payload = asyncio.run(github_data_extractor.github_extractor())

    print(f"\n📊 Repositories Processed Metadata Rows: {len(github_payload['metadatas'])}")
