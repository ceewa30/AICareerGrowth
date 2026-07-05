import sys
import os
from typing import List, Optional, Union
import math
# Resolve system paths for modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.VectorEmbedding import VectorEmbedding
from commons.DataValidation import UnifiedCareerReport
from CareerTrackerAgent import CareerTracker
from Gaurdrails.RetrievalEvaluator import RetrievalEvaluator
from Gaurdrails.GenerationGuardrails import GenerationGuardrails
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

db_path = Path(__file__).resolve().parent.parent / "chroma_storage"

# =====================================================================
# 2. THE RAG COMPARISON ENGINE CLASS
# =====================================================================
class RAGComparisonEngine:
    def __init__(self, database_path: str = db_path):
        # Initialize standard clients
        self.openai_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=str(database_path))

        # Configure embedding model synchronization helper
        self.openai_ef = embedding_functions.DefaultEmbeddingFunction()

        # Connect explicitly to your active collection framework
        self.collection = self.chroma_client.get_or_create_collection(
            name="developer_portfolios_v2",
            embedding_function=self.openai_ef
        )

        self.embedding_client = VectorEmbedding()
        self.tracker = CareerTracker()
        self.available_filters = self.tracker.get_all_available_filters()


    def generate_career_gap_report(self, target_role: str, user_profile_id: str) -> dict:
        """
        Executes dynamic vector similarity lookups for benchmarks and profiles using
        similarity_search, then feeds the structured context into OpenAI for roadmap extraction.
        """
        print(f"\n🔍 [RAG Step 1/3] Initiating context retrieval for target goal: '{target_role}'...")

        user_skills_list = []
        user_context_chunks = []
        user_context = "Candidate profile context is currently unavailable."

        evaluator = RetrievalEvaluator(confidence_floor=35.0)

        # =====================================================================
        # A. SEMANTIC RETRIEVAL OF USER PROFILE DATA (DOCX, LINKEDIN, GITHUB)
        # =====================================================================
        try:
            user_search_query = f"Technical skills inventory, programming languages, databases, and project history matching {target_role}"

            user_results = self.tracker.similarity_search(
                query_text=user_search_query,
                n_results=5,
                filter_source=self.available_filters
            )

            # 🚀 Send raw results block straight through your fixed evaluator method
            clean_user_docs, clean_user_metas = evaluator.evaluate_retrieval_payload(user_results)

            # Check against your clean validated collections rather than raw_docs[0]
            if clean_user_docs:
                print(f" -> Successfully extracted {len(clean_user_docs)} profile data layers via similarity search.")
                user_context_chunks = clean_user_docs

                # Extract your candidate capability string inventories out of clean metadatas
                for meta in clean_user_metas:
                    if meta and isinstance(meta, dict):
                        raw_skills = meta.get("skills_list", meta.get("primary_stack", meta.get("tech_stack", "")))
                        if raw_skills:
                            extracted_parts = [s.strip() for s in str(raw_skills).split(",") if s.strip()]
                            user_skills_list.extend(extracted_parts)

                user_skills_list = list(set(user_skills_list))
            else:
                user_context = "No relevant candidate profile chunks matching the query passed quality gates."

        except Exception as e:
            print(f"⚠️ Warning: Could not execute semantic profile search: {e}")

        if user_context_chunks:
            user_context = "\n---\n".join(user_context_chunks)

        # =====================================================================
        # B. SEMANTIC RETRIEVAL OF ROLE BENCHMARKS
        # =====================================================================
        market_skills_list = []
        benchmark_context = "No role benchmark data found."

        try:
            benchmark_results = self.tracker.similarity_search(
                query_text=f"Role: {target_role}. Core Required Technical Skills and Competencies blueprint requirements.",
                n_results=5,
                filter_source=["benchmark"]
            )

            raw_bench_docs = benchmark_results.get("documents")
            raw_bench_metas = benchmark_results.get("metadatas")

            benchmark_chunks = raw_bench_docs[0] if (raw_bench_docs and len(raw_bench_docs) > 0 and raw_bench_docs[0] is not None) else []
            benchmark_metas = raw_bench_metas[0] if (raw_bench_metas and len(raw_bench_metas) > 0 and raw_bench_metas[0] is not None) else []

            if benchmark_chunks:
                benchmark_context = "\n---\n".join([str(doc) for doc in benchmark_chunks if doc])

                for meta in benchmark_metas:
                    if meta and isinstance(meta, dict):
                        raw_market_string = meta.get("skills_list", meta.get("primary_stack", ""))
                        if raw_market_string:
                            market_parts = [m.strip() for m in str(raw_market_string).split(",") if m.strip()]
                            market_skills_list.extend(market_parts)

                # Cleanly deduplicate your global target baseline array
                market_skills_list = list(set(market_skills_list))
        except Exception as e:
            print(f"⚠️ Warning: Failed to execute semantic benchmark lookup: {e}")
            benchmark_context = "No role benchmark data found."

        # =====================================================================
        # C. NATIVE MATHEMATICAL DELTA PRE-COMPUTATION
        # =====================================================================
        user_skills_lower = {str(s).strip().lower() for s in user_skills_list if s}
        market_skills_lower = {str(m).strip().lower() for m in market_skills_list if m}


        lowercase_matches = user_skills_lower.intersection(market_skills_lower)

        ui_matching_skills = [s for s in market_skills_list if s and s.strip().lower() in lowercase_matches]
        missing_skills_detected = [m for m in market_skills_list if m and m.strip().lower() not in user_skills_lower]

        total_required = len(market_skills_lower)
        ready_percentage = (len(lowercase_matches) / total_required * 100) if total_required > 0 else 0.0
        ready_percentage = round(ready_percentage, 2)

        print(f"📊 Fixed In-Engine Analytics Metrics Summary:")
        print(f"  -> Extracted User Skills Count: {len(user_skills_lower)} ({user_skills_list})")
        print(f"  -> Extracted Market Skills Count: {len(market_skills_lower)} ({market_skills_list})")
        print(f"  -> Calculated Score: {ready_percentage}% | Gaps Found: {len(missing_skills_detected)}")

        if isinstance(benchmark_context, list):
            flat_benchmark_text = "\n---\n".join([str(doc).strip() for doc in benchmark_context if doc])
        else:
            flat_benchmark_text = str(benchmark_context).strip()

        # Fallback guard if the data retrieval layer returns an empty result set
        if not flat_benchmark_text or flat_benchmark_text == "No role benchmark data found.":
            flat_benchmark_text = f"Target Role Specification Template for {target_role} is currently unindexed. Leverage core system knowledge to extract standard milestones."

        print("🧠 [RAG Step 2/3] Context assembled. Dispatching structured prompt payload to OpenAI...")

        # =====================================================================
        # D. LLM ROADMAP SYNTHESIS GENERATION
        # =====================================================================
        rag_prompt = f"""
        You are an elite automated talent upskilling agent. Compare the Candidate's Profile Context against the retrieved Live Market Benchmarks to formulate a structural upskilling roadmap.

        [NATIVE PIPELINE METRICS (GROUND TRUTHS)]
        * Candidate Role Readiness Score: {ready_percentage}%
        * Mathematically Verified Missing Gaps: {missing_skills_detected}
        * Existing Strengths Aligned: {ui_matching_skills}

        [CONTEXT BLOCK A: CANDIDATE CORE BASELINE]
        {user_context}

        [CONTEXT BLOCK B: TARGET LIVE MARKET BENCHMARKS]
        {flat_benchmark_text}

        [EXPLICIT ASPIRATIONAL TARGET ROLE]
        {target_role}

        ### STRICT EXECUTION RULES:
        1. **Top-10 Hard Skill Limitation:** Identify and isolate exactly the **TOP 10** most critical, high-impact technical skills missing from the candidate's baseline that appear in the benchmarks. Do not output more or less than 10 items.
        2. **Zero-Hallucination Gap Analysis:** Restrict your selected top-10 items exclusively to the tools provided in the `{missing_skills_detected}` array. Do not invent missing tools or capabilities.
        3. **Deterministic Sequence & Formatting:** Arrange these top 10 items in a logical, step-by-step chronological progression path. Divide them cleanly across the three required progression phases below.
        4. **Context-Driven Bridging:** Explicitly show how the candidate can use their specific strengths listed in `{ui_matching_skills}` to master these 10 new gaps faster.
        5. **No Soft Skill Platitudes:** Focus 100% on hard engineering stack dependencies, system design patterns, and concrete industry credentials. Completely ignore management or soft skill advice.

        ### EXPECTED OUTPUT STRUCTURE:

        ## 📊 ROLE READINESS METRIC
        * **Current Match:** {ready_percentage}% readiness for the **{target_role}** track.

        ## 🎯 PHASED UP-SKILLING ROADMAP (TOP 10 ONLY)
        * **Phase 1 (Immediate Foundation):** Steps 1 to 3 out of your selected top 10 tools required to bridge the absolute baseline code layers.
        * **Phase 2 (Advanced Infrastructure):** Steps 4 to 7 out of your selected top 10 tools focused on cloud, data streams, and structural deployment reliability.
        * **Phase 3 (Architectural Mastery):** Steps 8 to 10 out of your selected top 10 design patterns or enterprise certifications (e.g., AWS Certified Solutions Architect, CKA, or specialized ML credentials) to secure seniority.

        ## ⚡ ACCELERATION STRATEGY
        * Clear, punchy breakdown of how their existing stack provides a mathematical unfair advantage to accelerate this pivot.
        """

        llm_report_object = None

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional technical engineering coach specializing in upskilling paths."},
                    {"role": "user", "content": rag_prompt}
                ],
                response_format=UnifiedCareerReport,
                temperature=0.1,
                top_p=0.9
            )

            print("✨ [RAG Step 3/3] Report generated successfully!")
            llm_report_object = completion.choices[0].message.parsed
            # return {
            #     "readiness_score": ready_percentage,
            #     "report_data": completion.choices[0].message.parsed
            # }

        except Exception as e:
            print(f"❌ Critical Completion Error: Failed to run OpenAI synthesis. Detail: {str(e)}")
            # return {
            #     "readiness_score": ready_percentage,
            #     "report_data": None
            # }
            llm_report_object = None

        guardrails = GenerationGuardrails()

        # Run the safety verification checks, locking in ground truths and evicting hallucinations
        final_sanitized_report = guardrails.verify_and_align_output(
            engine_readiness_score=ready_percentage,
            engine_missing_gaps=missing_skills_detected,
            raw_llm_payload=llm_report_object
        )

        # Return the clean, protected payload dictionary straight to your Gradio interface loop
        return final_sanitized_report


# =====================================================================
# 3. VERIFICATION RUNNER PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Initialize engine targeting your storage collection space
    engine = RAGComparisonEngine()

    print("\n--- Running Global Strategy Check ---")
    engine.similarity_search(
        query_text="Full-stack application deployment pipelines using AWS cloud computing frameworks or Docker",
        n_results=2
    )

    # 🚀 Example B: Targeted Lookup targeting ONLY your Seeded Industry Benchmarks
    print("\n--- Running Targeted Benchmark Audit Check ---")
    engine.similarity_search(
        query_text="Core technical skills engineering stack blueprints for Machine Learning and AI Engineers",
        n_results=1,
        source_type="benchmark" # Uses your built-in metadata where-clause tag checking
    )

    # Target User ID formatted cleanly to match your standard CareerTracker outputs
    # target_user_id = "profile_full_stack_developer"
    # desired_role = "AI Engineer / Data Science"

    # # Generate the RAG comparison report object
    # report = engine.generate_career_gap_report(
    #     target_role=desired_role,
    #     user_profile_id=target_user_id
    # )

    # # Clean console visualization dump
    # print("\n" + "="*60)
    # print("                     FINAL AGENT COMPARISON REPORT              ")
    # print("="*60)
    # print(f"Target Career Goal: {desired_role}\n")

    # print("DETECTED EXPERIENCE STRENGTHS:")
    # for strength in report.current_strengths_detected:
    #     print(f"  ✔ {strength}")

    # print("\nIDENTIFIED CRITICAL KNOWLEDGE GAPS:")
    # for gap in report.core_gaps_identified:
    #     print(f"  ✖ {gap}")

    # print("\nSTRUCTURED RECOMMENDATION TIMELINE ROADMAP:")
    # for idx, step in enumerate(report.curated_roadmap, 1):
    #     print(f"\n  Phase {idx}: {step.certification_or_course} ({step.provider})")
    #     print(f"    * Target Skills:      {', '.join(step.focus_skills_addressed)}")
    #     print(f"    * Timeline Estimate:  ~{step.duration_weeks} Weeks")
    #     print(f"    * Strategic Execution: {step.strategic_rationale}")
    # print("="*60 + "\n")
