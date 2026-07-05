import os
import re
import json
import chromadb
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# =========================================================================
# SCHEMA SECTION: STRICT PYDANTIC TRACKING MODEL CONSTRAINTS
# =========================================================================
class CuratedMilestone(BaseModel):
    week_number: int = Field(description="The sequential week number of the sprint timeline loop")
    focus_sub_skill: str = Field(description="The specific missing competency domain targeted this week")
    curated_resource: str = Field(description="The name of a specific, high-quality course or project repository link")
    hands_on_task: str = Field(description="A practical application challenge matching the user's learning style")
    time_investment_hours: float = Field(description="Estimated hours required to complete this task")

class StructuredTrajectoryReport(BaseModel):
    target_role: str = Field(description="The title of the destination career node objective")
    spatial_l2_distance: float = Field(description="The raw floating-point L2 geometric distance extracted from ChromaDB")
    trajectory_completion_score_percentage: float = Field(
        description="A calculated readiness score from 0.0 to 100.0 based on spatial proximity"
    )
    missing_technical_skills: List[str] = Field(description="List of specific tools, paradigms, or frameworks missing")
    attained_technical_skills: List[str] = Field(description="List of tools present in the candidate profile that match the target")
    weekly_schedule: List[CuratedMilestone] = Field(description="The sequential week-by-week educational roadmap matrix")
    target_certification_validation: str = Field(description="The optimal professional exam cert to validate this shift")


# =========================================================================
# CORE CORE PIPELINE AND ENGINE CORE PLATFORM CLASS
# =========================================================================
class AgenticCareerPipelineWithHistory:
    def __init__(self, db_path: str = "./chroma_pipeline_db", history_file: str = "career_trajectory_history.json"):
        """Initializes ChromaDB vector space storage and the JSON append tracking manager."""
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="career_trajectory_space",
            metadata={"hnsw:space": "l2"} # Explicit Euclidean L2 geometry settings
        )
        self.openai_client = OpenAI() # Pulls standard OPENAI_API_KEY environment token
        self.history_file = history_file

    # -------------------------------------------------------------------------
    # PIPELINE STEP 1: DATA INGESTION & EMBEDDING
    # -------------------------------------------------------------------------
    def ingest_trajectory_nodes(self, user_id: str, current_skills: List[str], target_role: str, target_jd: str):
        """Tokenizes, structures, and pushes user metrics vs job targets into ChromaDB."""
        clean_user_skills = [s.strip().lower() for s in current_skills if s.strip()]

        user_vector_text = (
            f"Operational Baseline Vector Profile.\n"
            f"User Identifier Node: {user_id}\n"
            f"Verified Technical Competencies: {', '.join(clean_user_skills)}"
        )

        target_vector_text = (
            f"Target Market Matrix Objective Profile.\n"
            f"Target Role Title: {target_role}\n"
            f"Core Functional Job Criteria Requirements: {target_jd}"
        )

        self.collection.upsert(
            ids=[f"user_{user_id}", f"target_{user_id}"],
            documents=[user_vector_text, target_vector_text],
            metadatas=[
                {"node_type": "user_present_state", "entity_id": user_id},
                {"node_type": "market_target_state", "entity_id": user_id, "role_title": target_role}
            ]
        )
        print(f"✅ [STEP 1 COMPLETE]: Synchronized vector state changes for '{user_id}' in ChromaDB.")

    # -------------------------------------------------------------------------
    # PIPELINE STEP 2: SEMANTIC GAP ANALYSIS
    # -------------------------------------------------------------------------
    def calculate_spatial_trajectory_gap(self, user_id: str) -> Dict[str, Any]:
        """Calculates strict spatial distance geometries between baseline and target nodes."""
        target_node = self.collection.get(ids=[f"target_{user_id}"], include=["documents", "metadatas"])
        if not target_node or not target_node.get("documents"):
            raise FileNotFoundError(f"Target vector coordinates missing for User ID: {user_id}")

        target_text_query = target_node["documents"][0]
        target_metadata = target_node["metadatas"][0]

        spatial_query_result = self.collection.query(
            query_texts=[target_text_query],
            n_results=2,
            where={"entity_id": user_id}
        )

        ids = spatial_query_result["ids"][0]
        distances = spatial_query_result["distances"][0]
        documents = spatial_query_result["documents"][0]

        user_node_index = ids.index(f"user_{user_id}") if f"user_{user_id}" in ids else None
        l2_spatial_distance = distances[user_node_index] if user_node_index is not None else 1.5
        user_profile_text = documents[user_node_index] if user_node_index is not None else ""

        print(f"📊 [STEP 2 COMPLETE]: Computed Spatial Trajectory L2 Distance: {l2_spatial_distance:.4f}")

        return {
            "spatial_l2_distance_delta": float(l2_spatial_distance),
            "user_vector_text": user_profile_text,
            "target_vector_text": target_text_query,
            "target_role": target_metadata.get("role_title", "Unknown Role")
        }

    # -------------------------------------------------------------------------
    # PIPELINE STEP 3: AGENTIC RECOMMENDATION EXECUTION (STRICT PYDANTIC OUTPUT)
    # -------------------------------------------------------------------------
    def execute_structured_routing_recommendation(self, space_metrics: Dict[str, Any]) -> StructuredTrajectoryReport:
        """Utilizes an LLM routing agent to parse metrics into a strict Pydantic model response."""
        raw_l2 = space_metrics['spatial_l2_distance_delta']
        # Normalized mapping calculation logic: as spatial L2 drops toward 0, readiness converges to 100%
        calculated_readiness = max(0.0, min(100.0, (1.5 - raw_l2) * 100.0))

        agent_prompt = f"""
        You are an AI Tech Career Director and Systems Architecture Validator.
        Evaluate the following ChromaDB vector metrics and populate the required tracking schema exactly.

        === VECTOR GEOMETRY TELEMETRY ===
        Calculated Spatial Vector Distance Delta (L2 metric): {raw_l2:.4f}
        Assigned Profile Readiness Baseline: {calculated_readiness:.1f}%

        === EXTRACTED VECTOR PROFILES ===
        PRESENT USER PROFILE:
        {space_metrics['user_vector_text']}

        TARGET TRJECTORY OBJECTIVE:
        Target Role: {space_metrics['target_role']}
        Target Job Description Criteria: {space_metrics['target_vector_text']}
        """

        response = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": agent_prompt}],
            response_format=StructuredTrajectoryReport,
            temperature=0.1
        )

        print("🤖 [STEP 3 COMPLETE]: Routing Agent evaluation validated and parsed successfully.")
        return response.choices[0].message.parsed

    # =========================================================================
    # HISTORY MODULE: TIME-SERIES FILE APPEND TRACKER
    # =========================================================================
    def log_trajectory_history(self, user_id: str, report: StructuredTrajectoryReport):
        """Appends active telemetry snapshot directly to a localized history log tracking file."""
        timestamp_key = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Prepare log snapshot object entry
        log_entry = {
            "timestamp": timestamp_key,
            "user_id": user_id,
            "target_role": report.target_role,
            "spatial_l2_distance": round(report.spatial_l2_distance, 4),
            "completion_score_percentage": round(report.trajectory_completion_score_percentage, 1),
            "missing_skills_count": len(report.missing_technical_skills),
            "attained_skills_count": len(report.attained_technical_skills)
        }

        # 2. Safely read existing file logs structure array
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as file:
                    history_data = json.load(file)
                    if not isinstance(history_data, list):
                        history_data = []
            except Exception:
                history_data = []
        else:
            history_data = []

        # 3. Append current step profile matrix node logs
        history_data.append(log_entry)

        with open(self.history_file, "w") as file:
            json.dump(history_data, file, indent=2)

        print(f"💾 [HISTORY LOGGED]: Appended progress data checkpoint to '{self.history_file}'.")
        self._display_historical_trend()

    def _display_historical_trend(self):
        """Prints a clean data summary trace reflecting trajectory changes."""
        try:
            with open(self.history_file, "r") as file:
                history = json.load(file)

            print("\n📈 [PROGRESS HISTORICAL METRIC TRACE]:")
            print(f"{'Timestamp':<20} | {'Target Role':<15} | {'L2 Distance':<12} | {'Completion %':<12}")
            print("-" * 68)
            for entry in history[-5:]: # View the last 5 updates
                print(f"{entry['timestamp']:<20} | {entry['target_role']:<15} | {entry['spatial_l2_distance']:<12.4f} | {entry['completion_score_percentage']:<12.1f}%")
                print("=" * 68)
        except Exception as e:
            print(f"Could not render timeline dashboard charts: {e}")

### Step 3: Run the Iterative Skill Logging Simulation Flow
"""
This script sets up a loop to simulate logging new skills over time, demonstrating how the system registers your progress as your profile coordinates move closer to your target goal.

```python """
if __name__ == "__main__":
    # Initialize Pipeline Engine instance setup
    pipeline = AgenticCareerPipelineWithHistory()

    user_id = "ceewa30"
    target_role = "AI Engineer"
    job_description = (
        "We need an AI Engineer to architect high-throughput model inference pipelines. "
        "Must have deep hands-on expertise building distributed LLM applications with LangChain "
        "and managing production scaling. This role directly coordinates with enterprise product "
        "stakeholders to align tech infrastructure with client business roadmaps."
    )

    # -------------------------------------------------------------------------
    # ITERATION 1: Initial Baseline Footprint
    # -------------------------------------------------------------------------
    print("\n🏁 --- RUNNING ITERATION 1: INITIAL BASELINE ---")
    skills_v1 = ["Python", "PHP", "MySQL", "HTML", "CSS", "AWS", "APIs"]

    pipeline.ingest_trajectory_nodes(user_id, skills_v1, target_role, job_description)
    metrics_v1 = pipeline.calculate_spatial_trajectory_gap(user_id)
    report_v1 = pipeline.execute_structured_routing_recommendation(metrics_v1)
    pipeline.log_trajectory_history(user_id, report_v1)

    # -------------------------------------------------------------------------
    # ITERATION 2: User acquires LangChain skills
    # -------------------------------------------------------------------------
    print("\n🚀 --- RUNNING ITERATION 2: NEW SKILL LOGGED [LANGCHAIN] ---")
    skills_v2 = ["Python", "PHP", "MySQL", "HTML", "CSS", "AWS", "APIs", "LangChain"]

    pipeline.ingest_trajectory_nodes(user_id, skills_v2, target_role, job_description)
    metrics_v2 = pipeline.calculate_spatial_trajectory_gap(user_id)
    report_v2 = pipeline.execute_structured_routing_recommendation(metrics_v2)
    pipeline.log_trajectory_history(user_id, report_v2)

    # -------------------------------------------------------------------------
    # ITERATION 3: User acquires Vector DB skills
    # -------------------------------------------------------------------------
    print("\n🏆 --- RUNNING ITERATION 3: NEW SKILL LOGGED [CHROMADB VECTOR STORES] ---")
    skills_v3 = ["Python", "PHP", "MySQL", "HTML", "CSS", "AWS", "APIs", "LangChain", "Vector Databases", "ChromaDB"]

    pipeline.ingest_trajectory_nodes(user_id, skills_v3, target_role, job_description)
    metrics_v3 = pipeline.calculate_spatial_trajectory_gap(user_id)
    report_v3 = pipeline.execute_structured_routing_recommendation(metrics_v3)
    pipeline.log_trajectory_history(user_id, report_v3)
