import os
import sys
from pathlib import Path
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from commons.DataValidation import DynamicRepoMetadata, READMEChunk
from commons.TargetSkill import TargetSkill
from commons.chunk import recursive_text_splitter
from core.VectorEmbedding import VectorEmbedding
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

class PDFDataExtractor:
    def __init__(self):
        self.client = OpenAI()
        self.target_skill = TargetSkill()
        self.username = "ceewa30"
        self.embedding_client = VectorEmbedding()

    def pdf_extractor(self):
    # 1. Fetch raw data from your target skill utility
        self.linkedin_data = self.target_skill.targetskill()
        readme_text = self.linkedin_data['linkedin_text']

        print("\n🚀 PIPELINE RETURN DATA RECEIVED:")
        print(f"🔹 Returned Role string: {self.linkedin_data['current_role']}")
        print(f"🔹 Total text count: {len(readme_text)} characters")
        print(f"🔹 Verified Skills : {self.linkedin_data['linkedin_skills']}")
        print(f"🔹 Summary content length: {self.linkedin_data['linkedin_summary']} characters")

        # 🚀 STEP A: Extract structured metadata using your OpenAI parse utility
        print(f"🤖 Dynamically extracting career profile metadata layers...")
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
        Analyze the following comprehensive professional profile and extract structured metadata features.
        PROFILE TEXT CONTENT:
        {readme_text}
        """

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=DynamicRepoMetadata  # Reuses your Pydantic extraction model
            )
            extracted_meta = completion.choices[0].message.parsed
        except Exception as e:
            print(f"⚠️ Metadata extraction extraction dropout. Falling back to basic parameters. Detail: {e}")
            # Dynamic fallback if API drops or rejects context parameters
            extracted_meta = DynamicRepoMetadata(
                author=self.linkedin_data.get('name', 'Unknown Candidate'),
                tech_stack=", ".join(list(self.linkedin_data['linkedin_skills'])),
                uses_machine_learning=False,
                algorithm_summary="N/A"
            )

        # 🚀 STEP B: Format the flat Chroma-compliant metadata base dictionary
        flat_profile_metadata = {
            "repository_name": "linkedin_profile_source",
            "author": extracted_meta.author,
            "primary_stack": extracted_meta.tech_stack,
            "ml_used": extracted_meta.uses_machine_learning,
            "algo_type": extracted_meta.algorithm_summary,
            "current_role": extracted_meta.current_role,
            "file_type": "linkedin"
        }

        # 2. Segment the raw profile down to 800 character window fragments
        text_chunks = recursive_text_splitter(readme_text, chunk_size=800, chunk_overlap=150)
        print(f" -> Generated {len(text_chunks)} chunk rows.")

        # 3. Generate 1536-dim OpenAI vector float arrays
        self.chunk_vectors = self.embedding_client.generate_batch_embeddings(text_chunks)

        # 🚀 STEP C: Loop and append explicit indexing properties for downstream storage arrays
        linkedin_documents = []
        linkedin_metadatas = []
        linkedin_ids = []
        linkedin_structured_chunks = []

        for idx, chunk_text in enumerate(text_chunks):
            row_id = f"linkedin_profile_chunk_{idx}"

            # Inject localized indexing positions unique to this chunk row
            chunk_specific_meta = flat_profile_metadata.copy()
            chunk_specific_meta["chunk_index"] = idx

            # Instantiate model to fulfill validation constraints
            chunk_obj = READMEChunk(
                chunk_id=row_id,
                chunk_index=idx,
                content=chunk_text,
                metadata=chunk_specific_meta,
                embedding=self.chunk_vectors[idx]
            )
            linkedin_structured_chunks.append(chunk_obj)

            # Prepare flat database layout buffers
            linkedin_documents.append(chunk_text)
            linkedin_metadatas.append(chunk_specific_meta)
            linkedin_ids.append(row_id)
        # print(f"LinkedIn Metadata: {chunk_specific_meta}")
        # Pack everything into an organized output container dictionary payload
        self.linkedin_payload = {
            "raw_data": self.linkedin_data,
            "structured_chunks": linkedin_structured_chunks,
            "documents": linkedin_documents,
            "metadatas": linkedin_metadatas,
            "ids": linkedin_ids,
            "embeddings": self.chunk_vectors
        }

        return self.linkedin_payload


if __name__=="__main__":
    extractor = PDFDataExtractor()

    linkedin_payload = extractor.pdf_extractor()

    print(f"\n📊 Repositories Processed Metadata Rows: {len(linkedin_payload['metadatas'])}")
