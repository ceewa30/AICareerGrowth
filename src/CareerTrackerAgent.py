import sys
import os
import yaml
from pathlib import Path
from typing import Optional, List, Union
import math
# Resolve system paths for modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.VectorEmbedding import VectorEmbedding
import asyncio
import chromadb
from chromadb.utils import embedding_functions

from core.PDFDataExtractor import PDFDataExtractor
from core.GitHubDataExtractor import GitHubDataExtractor
from commons.DocumentExtractor import DocumentExtractor

class CareerTracker:
    def __init__(self):
        self.pdf_data_extractor = PDFDataExtractor()
        self.github_data_extractor = GitHubDataExtractor()
        self.doc_data_extractor = DocumentExtractor()
        self.embedding_client = VectorEmbedding()

        # Resolve path data variables safely
        self.db_storage_path = Path(__file__).resolve().parent.parent / "chroma_storage"
        self.db_storage_path.mkdir(parents=True, exist_ok=True)

        # 1: Explicitly cast Path object to string for persistent setup safety
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_storage_path))

        # 2: Swapped model to OpenAI to prevent critical database model initialization collision crashes
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        # 3: Aligned collection name key exactly with your RAG engine's tracking target space
        self.collection = self.chroma_client.get_or_create_collection(
            name="developer_portfolios_v2",
            embedding_function=self.embedding_function
        )

        # Automatically verify and run baseline repository seeding
        self._seed_market_standard()

    def _seed_market_standard(self):
        """Seeds the vector DB with industry benchmarks for role hierarchies and skill requirements."""
        config_path = Path(__file__).resolve().parent.parent / "config" / "market_standards.yaml"
        if not config_path.exists():
            print(f"❌ Configuration not found at: {config_path}")
            return

        with open(config_path, "r", encoding="utf-8") as stream:
            try:
                config_data = yaml.safe_load(stream)
                standards = config_data.get("roles", {})
            except yaml.YAMLError as exc:
                print(f"❌ Error compiling configuration data layout: {exc}")
                return

        print(f"🗄️ Seeding Vector DB with {len(standards)} decoupled rulesets...")
        for role, requirements in standards.items():
            raw_skills = requirements.get("skills", [])
            if isinstance(raw_skills, dict):
                all_tools = []
                for category_list in raw_skills.values():
                    if isinstance(category_list, list):
                        all_tools.extend(category_list)
                skills_string = ", ".join(all_tools)
            elif isinstance(raw_skills, list):
                skills_string = ", ".join(raw_skills)
            else:
                flat_list = requirements.get("flat_skills_list", [])
                skills_string = ", ".join(flat_list) if flat_list else "N/A"

            document_content = (
                f"Role: {role}. Required Experience: {requirements.get('experience', 'N/A')}. "
                f"Core Required Technical Skills and Competencies: {skills_string}"
            )

            metadata = {
                "role": role,
                "experience_bracket": requirements.get('experience', 'N/A'),
                "type": "market_benchmark",
                "skills_list": str(skills_string)
            }

            try:
                safe_id = f"benchmark_{role.lower().replace(' ', '_').replace('/', '_')}"
                vector_results = self.embedding_client.generate_batch_embeddings([document_content])

                if not vector_results:
                    print(f"⚠️ Warning: Failed to generate embedding vector array for role: {role}")
                    continue

                vector_embedding = vector_results[0]
                # .upsert ensures infinite safe operational runs without duplicates or crashes
                self.collection.upsert(
                    ids=[safe_id],
                    metadatas=[metadata],
                    embeddings=[vector_embedding],
                    documents=[document_content]
                )
                print(f"✅ Seeded benchmark entity: {role}")
            except AttributeError:
                print(f"⚠️ Vector collection uninitialized. Processed item memory structure: {role}")

    def skill_combine(self):
        """Processes resume text extraction payloads and maps them securely to your data vectors."""
        print("\n🚀 Initiating Skill Combination Ingestion Pipeline...")

        try:
            # =====================================================================
            # 1. ORCHESTRATE ASYNC TASKS CONCURRENTLY IN A SINGLE LOOP
            # =====================================================================
            async def run_async_extractors():
                print("🔄 Fetching LinkedIn PDF data and GitHub repositories concurrently...")

                # Fire both heavy network/API extractors at the exact same time
                linkedin_task = self.pdf_data_extractor.pdf_extractor()
                github_task = self.github_data_extractor.github_extractor()
                docx_task = self.doc_data_extractor.process_file_upload()

                # Concurrently await both results over a single event loop execution
                return await asyncio.gather(linkedin_task, github_task, docx_task)

            try:
                # CALL ASYNCIO.RUN EXACTLY ONCE FOR ALL ASYNC EXTRACTIONS
                self.linkedin_payload, self.github_payload, self.docx_payload = asyncio.run(run_async_extractors())
            except Exception as async_err:
                print(f"❌ Extraction Error: Concurrent async extraction worker crashed. Detail: {str(async_err)}")
                raise async_err

            # =====================================================================
            # 2. SYNCHRONIZE LINKEDIN DOCUMENT ROWS INTO THE COLLECTION
            # =====================================================================
            print("\n========================================================")
            print("            LinkedIn PDF INGESTION RAW JSON DATA        ")
            print("========================================================")

            if self.linkedin_payload.get("documents"):
                print(f"📥 Loading {len(self.linkedin_payload['documents'])} LinkedIn profile segments into Vector database...")
                try:
                    self.collection.upsert(
                        ids=self.linkedin_payload["ids"],
                        embeddings=self.linkedin_payload["embeddings"],
                        metadatas=self.linkedin_payload["metadatas"],
                        documents=self.linkedin_payload["documents"]
                    )
                    print("✅ LinkedIn profile segment storage synchronization finalized.")
                except Exception as db_err:
                    print(f"❌ Database Error: Failed to upsert LinkedIn rows to ChromaDB. Detail: {str(db_err)}")
                    raise db_err
            else:
                print("⚠️ Ingestion Warning: No readable LinkedIn profile segments found to sync.")

            # =====================================================================
            # 3. SYNCHRONIZE GITHUB REPOSITORY DOCUMENT ROWS INTO THE COLLECTION
            # =====================================================================
            print("\n========================================================")
            print("           GitHub REPOSITORY INGESTION RAW JSON DATA    ")
            print("========================================================")

            if self.github_payload.get("documents"):
                print(f"📥 Bulk loading {len(self.github_payload['documents'])} GitHub chunk streams into Vector database...")
                try:
                    self.collection.upsert(
                        ids=self.github_payload["ids"],
                        embeddings=self.github_payload["embeddings"],
                        metadatas=self.github_payload["metadatas"],
                        documents=self.github_payload["documents"]
                    )
                    print(f"✅ Successfully updated {len(self.github_payload['ids'])} GitHub items in ChromaDB.")
                except Exception as db_err:
                    print(f"❌ Database Error: Failed to upsert GitHub repositories to ChromaDB. Detail: {str(db_err)}")
                    raise db_err
            else:
                print("⚠️ Ingestion Warning: No GitHub data layers collected to push to store.")

            # =====================================================================
            # 4. PROCESS AND EXTRACT DOCX FILE LAYERS (PURE SYNCHRONOUS FLOW)
            # =====================================================================
            print("\n========================================================")
            print("             Resume Docx INGESTION RAW JSON DATA        ")
            print("========================================================")

            # =====================================================================
            # 5. SYNCHRONIZE PROFILE DOCUMENT ROWS INTO THE COLLECTION
            # =====================================================================
            if self.docx_payload.get("documents") and len(self.docx_payload["documents"]) > 0:
                print(f"📥 Loading {len(self.docx_payload['documents'])} unique Docx profile segments into Vector database...")
                try:
                    self.collection.upsert(
                        ids=self.docx_payload["ids"],
                        embeddings=self.docx_payload["embeddings"],
                        metadatas=self.docx_payload["metadatas"],
                        documents=self.docx_payload["documents"]
                    )
                    print("✅ Document profile segment storage synchronization finalized.")
                except Exception as db_err:
                    print(f"❌ Database Error: Failed to upsert Docx rows to ChromaDB. Detail: {str(db_err)}")
                    raise db_err
            else:
                print("⚠️ Ingestion Warning: No readable Docx profile segments found to sync.")

            print("\n✓ Ingestion Synchronization Layers Executed Successfully!")

        except KeyError as ke:
            print(f"❌ Payload Schema Error: Missing internal dictionary tracking key mapping. Detail: {str(ke)}")
        except Exception as global_err:
            print(f"❌ Critical Pipeline Failure: Ingestion crashed due to an unhandled system exception.")
            print(f" Detail: {str(global_err)}")


    def get_all_available_filters(self) -> dict:
        """
        Scans the collection, isolates stored metadata fields,
        and returns a summary of all unique keys available for filtering.
        """
        print("\n🔍 [Metadata Discovery] Scanning collection to map out active filter attributes...")

        try:
            # Retrieve the metadata blocks for all rows in the collection
            records = self.collection.get(include=["metadatas"])
            metadatas = records.get("metadatas") or []

            if not metadatas:
                print("⚠️ The database collection is empty. Run your ingestion loops first.")
                return {"file_types": [], "repository_names": [], "types": []}

            # Use Python sets to isolate unique, distinct string occurrences
            unique_file_types = set()
            unique_repos = set()
            unique_types = set()

            for meta in metadatas:
                if not meta:
                    continue
                # Extract tracking fields mapped across your different parsers
                if "file_type" in meta:
                    unique_file_types.add(meta["file_type"])
                if "repository_name" in meta:
                    unique_repos.add(meta["repository_name"])
                if "type" in meta:
                    unique_types.add(meta["type"])

            # Package the metrics for systemic visibility
            available_filters = {
                "file_types": list(unique_file_types),
                "repository_names": list(unique_repos),
                "types": list(unique_types)
            }

            return available_filters['file_types']

        except Exception as e:
            print(f"❌ Discovery Fault: Failed to parse available filter properties. Detail: {str(e)}")
            return {}


    def similarity_search(self, query_text: str, n_results: int = 3, filter_source: Optional[Union[str, List[str]]] = None) -> dict:
        """
        Executes a semantic vector query against your integrated ChromaDB records.
        Automatically converts query strings to embeddings via the bound OpenAI function.
        """
        print(f"\n🔍 [Similarity Search] Querying database for: '{query_text[:60]}...'")

        # 1. Base query configuration mapping
        query_kwargs = {
            "query_texts": [query_text],
            "n_results": n_results
        }

        if filter_source:
            clean_sources = [str(s).lower().strip() for s in filter_source]

            file_type_targets = []
            repository_name_targets = []
            type_targets = [] # 🌟 ADD THIS: To capture global category types

            for src in clean_sources:
                if src == "github":
                    file_type_targets.append("github")
                elif src == "linkedin":
                    repository_name_targets.append("linkedin_profile_source")
                elif src in ["docx", "resume"]:
                    repository_name_targets.append("docx_resume_source")
                elif src == "benchmark":
                    # 🌟 MATCHING THE SEED METADATA KEY VALUE:
                    type_targets.append("market_benchmark")

            # Construct your ChromaDB multi-clause query using logical $or
            or_clauses = []
            if file_type_targets:
                or_clauses.append({"file_type": {"$in": file_type_targets}})
            if repository_name_targets:
                or_clauses.append({"repository_name": {"$in": repository_name_targets}})
            if type_targets:
                # 🌟 INJECTS EXPLICIT COMPLIANCE FOR YOUR SEEDED RECORDS
                or_clauses.append({"type": {"$in": type_targets}})

            if len(or_clauses) == 1:
                query_kwargs["where"] = or_clauses[0]
            elif len(or_clauses) > 1:
                query_kwargs["where"] = {"$or": or_clauses}


        try:
            # 3. Fire the vector space calculation lookup transaction
            results = self.collection.query(**query_kwargs)

            # 4. Unpack elements safely for a clean terminal report summary dump
            documents = results.get("documents", [[]])[0] if results.get("documents") else []
            ids = results.get("ids", [[]])[0] if results.get("ids") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []

            print("\n========================================================")
            print("              VECTOR RETRIEVAL SEARCH MATCHES            ")
            print("========================================================")

            if not documents:
                print("⚠️ Zero records matched your semantic search constraints.")
                return results

            for rank in range(len(documents)):
                # Convert raw distance to a human-readable match confidence scale metric string
                raw_dist = distances[rank] if rank < len(distances) else 0.5

                # Transform standard L2 vector distances into legible percentage values
                confidence_score = max(0.0, min(100.0, round(math.exp(-raw_dist) * 100, 2)))

                meta_block = metadatas[rank] if rank < len(metadatas) else {}
                source_tag = meta_block.get("repository_name", meta_block.get("file_type", "docx_resume_source"))

                print(f" 🎯 [Rank #{rank + 1} - Confidence Match: {confidence_score}%] (ID: {ids[rank]})")
                print(f" 📂 Source Origin Domain: {source_tag}")
                print(f" 📄 Content Snippet Text: {documents[rank][:140].strip()}...")
                print("-" * 56)

            return results

        except Exception as e:
            print(f"❌ Database Query Fault: Failed to execute similarity search. Detail: {str(e)}")
            return {}

if __name__ == "__main__":
    tracker = CareerTracker()
    tracker.skill_combine()

    print(tracker.get_all_available_filters())

    print("\n--- Running Global Career Asset Query Test ---")
    tracker.similarity_search(
        query_text="Experienced developer building cloud backend pipelines with Python and Docker infrastructure orchestration",
        n_results=2
    )

    # 🚀 Run a targeted audit querying specifically inside your Docx Resume chunks only
    print("\n--- Running Target Resume Query Test ---")
    tracker.similarity_search(
        query_text="Comprehensive history of programming languages, hands-on framework exposure, pipeline development, and deployed architecture infrastructure",
        n_results=4,
        filter_source=tracker.get_all_available_filters() # Uses your new metadata keyword query parser
    )

    tracker.similarity_search(
    query_text="AI Engineer",
    n_results=1,
    filter_source="benchmark"
)
