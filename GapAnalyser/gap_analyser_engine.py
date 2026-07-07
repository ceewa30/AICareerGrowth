import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import yaml
from pathlib import Path
import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from commons.PDFSkillLinkedInExtractor import PDFSkillExtractor

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

class GapAnalyserEngine:
    def __init__(self):
        # Initialize an in-memory Storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_workspace")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="role_hierarchy_matrix",
            embedding_function=self.embedding_function
        )
        self._seed_market_standard()

    def _seed_market_standard(self):
        """Seeds the vector DB with industry benchmarks for role hierarchies and skill requirements."""
        # Resolve path relative to this file to handle workspace movements safely
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
            # Safe skill resolution wrapper to handle both flat lists and nested dictionaries
            if not requirements:
                continue

            raw_skills = requirements.get("skills", [])

            if isinstance(raw_skills, dict):
                # Scenario A: Handle nested dictionary data structures (e.g., AI Engineer, Full Stack)
                all_tools = []
                for category_list in raw_skills.values():
                    if isinstance(category_list, list):
                        all_tools.extend(category_list)
                skills_string = ", ".join(all_tools)
            elif isinstance(raw_skills, list):
                # Scenario B: Handle standard flat list structures (e.g., Software Engineer, Tech Lead)
                skills_string = ", ".join(raw_skills)
            else:
                # Fallback security check if flat_skills_list is present at root tier
                flat_list = requirements.get("flat_skills_list", [])
                skills_string = ", ".join(flat_list) if flat_list else "N/A"

            # Assemble vector engine payload text document
            document_content = (
                f"Role: {role}. Required Experience: {requirements.get('experience', 'N/A')}. "
                f"Core Required Technical Skills and Competencies: {skills_string}"
            )

            metadata = {
                "role": role,
                "experience_bracket": requirements.get('experience', 'N/A'),
                "type": "market_benchmark"
            }

            try:
                # Generate a clean, URL-safe vector embedding identifier token
                safe_id = f"benchmark_{role.lower().replace(' ', '_').replace('/', '_')}"

                self.collection.add(
                    documents=[document_content],
                    metadatas=[metadata],
                    ids=[safe_id]
                )
                print(f"✅ Seeded benchmark entity: {role}")
            except AttributeError:
                print(f"⚠️ Vector collection uninitialized. Processed item memory structure: {role}")

    def generate_gap_analysis(self, user_id: str, target_role: str) -> dict:
        """
        Connects mapped profile assets to target criteria to isolate structural
        technical and strategic career gaps.
        """
        print(f"\n🧠 Initiating Semantic Gap Analysis for User '{user_id}' -> Target Role: '{target_role}'...")

        # Pre-define fallback tracking variables to prevent NameErrors if blocks fail early
        linkedin_skills = []
        profile_doc = ""
        current_role = "Developer"
        matched_role_name = target_role
        experience_bracket = "N/A"
        recommendation = "Analysis incomplete due to profile extraction errors."
        missing_skills = []
        attained_skills = []

        # 1. Fetch the user's integrated profile from the collection
        try:
            user_data = self.collection.get(
                ids=[f"user_{user_id}"],
                include=["documents", "metadatas"]
            )

            if not user_data.get("documents") or len(user_data["documents"]) == 0:
                raise Exception(f"Profile for user {user_id} not integrated into Vector DB yet.")

            # Extract user skills from flattened metadata string
            user_metadata_list = user_data.get("metadatas", [])
            # print(f"Metadata {user_metadata_list}")
            # FIX A: Comprehensive fallback for missing or completely unpopulated metadata blocks
            if user_metadata_list and isinstance(user_metadata_list, list) and user_metadata_list[0] is not None:
                user_metadata = user_metadata_list[0]

            else:
                user_metadata = {}

            profile_doc = user_data["documents"][0].lower() if user_data["documents"] else ""

            # FIX B: Intercept situations where 'linkedin_skills' key exists but contains an explicit None value
            linkedin_skills_raw = user_metadata.get("linkedin_skills", "")
            linkedin_skills_str = linkedin_skills_raw if linkedin_skills_raw is not None else ""

            # Dynamically parse the skills sequence regardless of whether it arrives as list or string
            if isinstance(linkedin_skills_str, list):
                linkedin_skills = [s.strip().lower() for s in linkedin_skills_str if s.strip()]
            else:
                linkedin_skills = [s.strip().lower() for s in linkedin_skills_str.split(",") if s.strip()]
            print(f"LinkedIn Skills {linkedin_skills}")
        except Exception as e:
            return {"error": f"Failed to retrieve user profile: {str(e)}"}

        # 2. Query the database using semantic search to find the closest matching market benchmark
        try:
            benchmark_result = self.collection.query(
                query_texts=[f"Role: {target_role}"],
                n_results=1,
                where={"type": "market_benchmark"}
            )

            if not benchmark_result.get("documents") or len(benchmark_result["documents"]) == 0 or len(benchmark_result["documents"][0]) == 0:
                raise Exception(f"No market benchmark found matching role: {target_role}")

            benchmark_doc = benchmark_result["documents"][0][0]

            # FIX C: Add type safety layout fallback parameters for the benchmark metadata layer
            benchmark_meta_list = benchmark_result.get("metadatas", [])

            benchmark_meta = benchmark_meta_list[0][0] if (benchmark_meta_list and benchmark_meta_list[0]) else {}

            if benchmark_meta is None:
                benchmark_meta = {}

            matched_role_name = benchmark_meta.get("role", target_role)

            experience_bracket = benchmark_meta.get("experience_bracket", "N/A")

        except Exception as e:
            return {"error": f"Failed to cross-reference market benchmark: {str(e)}"}

        # 3. Extract core benchmark skills from the matching database document
        benchmark_skills = []
        search_key = "technical skills and competencies:"
        raw_doc_text = benchmark_doc if isinstance(benchmark_doc, str) else benchmark_doc[0]

        if search_key in raw_doc_text.lower():
            skills_part = raw_doc_text.lower().split(search_key)[1]
            # Split the comma-separated skills into a clean python list
            benchmark_skills = [s.strip() for s in skills_part.split(",") if s.strip()]

        for b_skill in benchmark_skills:
            if b_skill in linkedin_skills or b_skill in profile_doc:
                attained_skills.append(b_skill.title())
            else:
                missing_skills.append(b_skill.title())
        print(f"Attained Skill {attained_skills}")

        # 4. Generate career path logic recommendation context
        current_role = user_metadata.get("current_role", "Developer")
        current_role = current_role.title() if current_role else "Developer"

        client = OpenAI()

        if missing_skills:
            recommendation = f"To transition successfully from '{current_role}' to '{matched_role_name}', you need to acquire active experience in: {', '.join(missing_skills)}."
        else:
            recommendation = f"Your technical profile cleanly meets or exceeds all tracked database benchmarks for '{matched_role_name}'!"

        # prompt = f"""
        #     You are an Executive Technology Career Coach. Analyze the user's mapped professional profile against their target trajectory role.

        #     --- USER PROFILE DATA ---
        #     Current Title: {current_role}
        #     GitHub Footprint: {', '.join(profile['github_top_languages'])} ({profile['github_repositories_count']} repos)
        #     LinkedIn Endorsed Skills: {attained_skills}
        #     Experience Summary: {profile['experience_summary']}

        #     --- TARGET TARGET TRAJECTORY ---
        #     Target Role: {matched_role_name}

        #     --- OBJECTIVE ---
        #     Identify the absolute missing technical frameworks, architectural capabilities, and leadership/management gaps keeping them from this role.

        #     Format your response in a highly scannable layout:
        #     ### 🎯 Mapped Trajectory Delta: {current_role} ➔ {matched_role_name}

        #     **1. HARD SKILL & ARCHITECTURE GAPS**
        #     * [Provide 1-2 bullet points specifying advanced tech, architecture, or paradigms missing, e.g., System Design at scale, Vector databases, etc.]

        #     **2. LEADERSHIP & STRATEGIC GAPS**
        #     * [Provide 1-2 bullet points specifying missing human, product, or organizational capabilities, e.g., Stakeholder Management, Budgeting, Mentorship.]

        #     **3. THE IMMEDIATE CORRECTION PLAN**
        #     * [Actionable execution task]
        #     """


        return {
            "user_id": user_id,
            "current_role": current_role,
            "target_role": matched_role_name,
            "experience_bracket_required": experience_bracket,
            "matching_skills_detected": attained_skills,
            "missing_skills_gap": missing_skills,
            "actionable_recommendation": recommendation
        }



    # Wrap function parameters dynamically to safely digest any structural inputs
    def integrate_external_profile(self, **kwargs):
        """Integrates external profile data into the vector database using a search-optimized kwargs approach."""

        # Safely pop variables out of the dictionary with smart string/list fallbacks
        user_id = kwargs.get("user_id", "unknown_user")
        linkedin_skills = kwargs.get("linkedin_skills", [])
        github_repos = kwargs.get("github_repos", [])
        current_role = kwargs.get("current_role", "none")
        linkedin_summary = kwargs.get("linkedin_summary", "none")
        github_skills = kwargs.get("github_skills", [])

        print(f"GitHub repos {github_repos}")

        if isinstance(linkedin_skills, str):
            raw_li_skills = [s.strip() for s in linkedin_skills.split(",") if s.strip()]
        elif isinstance(linkedin_skills, (list, set)):
            raw_li_skills = linkedin_skills
        else:
            raw_li_skills = []

        # Dynamically extract plural or singular forms to protect against variable drift
        github_skills = kwargs.get("github_skills", kwargs.get("github_skill", []))

        # clean_skills = [s.strip().lower() for s in raw_li_skills if s.strip()]

        clean_repos = [r.strip().lower() for r in github_repos if r.strip()]
        # print(f"github repop :{clean_repos}")

        role = current_role.strip().lower() if current_role.strip() else "none"

        if isinstance(github_skills, dict):
            raw_skills = github_skills.keys()
        elif isinstance(github_skills, (list, set)):
            raw_skills = github_skills
        else:
            raw_skills = []

        clean_ln_skills = [str(s).strip().lower() for s in raw_li_skills if str(s).strip()]
        clean_gh_skills = [str(s).strip().lower() for s in raw_skills if str(s).strip()]

        merged_skills_lowercase = list(set(clean_gh_skills + clean_ln_skills))
        # print(f"skill unique: {clean_gh_skills}")
        seen_skills = {}
        for s in merged_skills_lowercase:
            display_name = s.title()

            if s not in seen_skills:
                seen_skills[s] = display_name

        all_skills_unique = list(seen_skills.values())

        # 🌟 FIXED: Format the clean whole-words array into bracket-encapsulated metadata strings
        skills_bracketed_string = f"[{','.join(all_skills_unique)}]"
        print(f"Skills details : {skills_bracketed_string}")

        # 4. Export as a unified comma-separated string block
        skills_list = ", ".join(all_skills_unique)

        profile_text = (
            f"Professional Profile.\n"
            f"Current Role: {role}\n"
            f"Skills: {skills_bracketed_string}\n"
            f"GitHub Repositories: {', '.join(clean_repos)}\n"
            f"LinkedIn Summary: {linkedin_summary}"
        )

        metadata = {
            "user_id": str(user_id),
            "linkedin_summary": linkedin_summary,
            "skills": skills_list,
            "github_repos": ", ".join(clean_repos),
            "current_role": role,
            "type": "external_profile"
        }
        # print(f"Metadata: {metadata}")
        self.collection.upsert(
            ids=[f"user_{user_id}"],
            documents=[profile_text],
            metadatas=[metadata]
        )
        print(f"✅ External profile for user {user_id} integrated successfully {skills_list} mapping.")



    def analyze_stored_profile_gap(self, user_id: str, target_role: str, target_job_description: str) -> str:
        """
        Retrieves the stored profile from ChromaDB using metadata filters,
        and runs a comparative AI gap analysis against a target job role.
        """
        print(f"\n🧠 Initiating Semantic Gap Analysis for User '{user_id}' -> Target Role: '{target_role}'...")
        matched_role_name = target_role
        job_description = target_job_description
        experience_bracket = "N/A"
        recommendation = "Analysis incomplete due to profile extraction errors."
        missing_skills = []
        attained_skills = []
        benchmark_skills = []

        # 1. Fetch the exact user document from ChromaDB using a metadata filter
        try:
            result = self.collection.get(
                ids=[f"user_{user_id}"],
                include=["documents", "metadatas"]
            )

            if not result or not result.get("documents") or len(result["documents"]) == 0:
                raise Exception(f"Profile for user {user_id} not found in ChromaDB memory storage.")

            user_metadata_list = result.get("metadatas", [])
            print(f"User metadata {user_metadata_list}")

            if user_metadata_list and isinstance(user_metadata_list, list) and user_metadata_list[0] is not None:
                user_metadata = user_metadata_list[0]
            else:
                user_metadata = {}

            profile_text = ""

            # Extract the stored text framework we engineered in previous steps
            profile_text = result["documents"][0].lower() if result["documents"] else ""

            skills_raw = user_metadata.get("skills", "")
            str_raw = skills_raw.replace(", ", "")

            raw_list = [s.strip().lower().replace("-", " ") for s in str_raw.split(",") if s.strip()]

            print(f"Skills details : {raw_list}")
        except Exception as e:
            return {"error": f"Failed to retrieve user profile: {str(e)}"}


        try:
            benchmark_result = self.collection.query(
                query_texts=[f"Role: {target_role}"],
                n_results=1,
                where={"type": "market_benchmark"}
            )

            print(f"Traget role: {matched_role_name}")

            documents = benchmark_result.get("documents") or []
            if not documents or len(documents) == 0 or len(documents[0]) == 0:
                raise Exception(f"No market benchmark found matching role: {target_role}")

            first_doc_entry = documents[0]

            raw_doc_text = first_doc_entry[0] if isinstance(first_doc_entry, list) else first_doc_entry
            print(f"Benchmark document content: {raw_doc_text}")

            benchmark_meta_list = benchmark_result.get("metadatas", [])

            benchmark_meta = benchmark_meta_list[0][0] if (benchmark_meta_list and benchmark_meta_list[0]) else {}
            if benchmark_meta is None:
                benchmark_meta = {}

            # matched_role_name = benchmark_meta.get("role", target_role)
            experience_bracket = benchmark_meta.get("experience_bracket", "N/A")

        except Exception as e:
            return {"error": f"Failed to cross-reference market benchmark: {str(e)}"}

        # 3. Safe Benchmark Extraction Loop

        search_key = "technical skills and competencies:"
        attained_skills = []
        missing_skills = []

        if search_key in raw_doc_text.lower():
            skills_part = raw_doc_text.lower().split(search_key)[1]

            raw_skills_list = [s.strip() for s in skills_part.replace("\n", ",").split(",") if s.strip()]
            # Clean up hyphens to match the processed user profile list structure exactly
            benchmark_skills = [s.strip().lower().replace("-", " ") for s in raw_skills_list]

        user_skills_clean = [str(item).lower().strip() for item in raw_list]
        profile_text_lower = profile_text.lower()

        for b_skill in benchmark_skills:
            # Check both the parsed whole-word list AND the dense profile document block
            if b_skill in user_skills_clean or f" {b_skill} " in f" {profile_text_lower} ":
                attained_skills.append(b_skill.title())
            else:
                missing_skills.append(b_skill.title())

        print(f"🎯 Attained Skills List Map: {attained_skills}")

        # 4. Generate career path logic recommendation context
        current_role = user_metadata.get("current_role", "")

        current_role = current_role.title() if current_role else "Developer"

        if missing_skills:
            recommendation = f"To transition successfully from '{current_role}' to '{matched_role_name}', you need to acquire active experience in: {', '.join(missing_skills)}."
        else:
            recommendation = f"Your technical profile cleanly meets or exceeds all tracked database benchmarks for '{matched_role_name}'!"


        # linkedin_summary = user_metadata.get("linkedin_summary", "")
        # github_repos = user_metadata.get("github_repos", "")
        # # 2. Initialize OpenAI to perform the delta analysis
        # client = OpenAI()

        # prompt = f"""
        #     You are an Executive Technology Career Coach. Analyze the user's mapped professional profile against their target trajectory role.

        #     --- USER PROFILE DATA ---
        #     Current Title: {current_role}
        #     GitHub Footprint: {github_repos} repos
        #     Skills: {attained_skills}
        #     Profile Summary: {linkedin_summary}

        #     --- TARGET TARGET TRAJECTORY ---
        #     Target Role: {matched_role_name}
        #     Experience required: {experience_bracket}

        #     --- OBJECTIVE ---
        #     Identify the absolute missing technical frameworks, architectural capabilities, and leadership/management gaps keeping them from this role.

        #     Format your response in a highly scannable layout:
        #     ### 🎯 Mapped Trajectory Delta: {current_role} ➔ {matched_role_name}

        #     ### {recommendation}

        #     **1. HARD SKILL & ARCHITECTURE GAPS**
        #     * [Provide 1-2 bullet points specifying advanced tech, architecture, or paradigms missing, e.g., System Design at scale, Vector databases, etc.]

        #     **2. LEADERSHIP & STRATEGIC GAPS**
        #     * [Provide 1-2 bullet points specifying missing human, product, or organizational capabilities, e.g., Stakeholder Management, Budgeting, Mentorship.]

        #     **3. THE IMMEDIATE CORRECTION PLAN**
        #     * [Actionable execution task]
        #     """

        # # prompt = f"""
        # # You are an AI Executive Tech Recruiter and Systems Design Expert.
        # # Compare the following Candidate Profile with the Target Job Framework.

        # # === CANDIDATE PROFILE MATRIX ===
        # # {profile_text}

        # # === TARGET JOB CRITERIA ===
        # # Target Role: {target_role}
        # # Job Description: {target_job_description}

        # # === REQUIRED OUTPUT SPECIFICATION ===
        # # Analyze the trajectory delta. Output your findings using this exact layout:

        # # ### 📊 Trajectory Gap Analysis: {target_role}

        # # **1. HARD SKILLS & INFRASTRUCTURE DELTA**
        # # * [Identify missing technical frameworks, languages, or architectural tools]

        # # **2. STRATEGIC & LEADERSHIP DELTA**
        # # * [Identify missing human management, stakeholder alignment, or product metrics experience]

        # # **3. ACTIONABLE PREPARATION ROADMAP**
        # # * [Provide 1 immediate project or task the user should start to bridge the gap]
        # # """

        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.2
        # )

        # return response.choices[0].message.content
        return "success"



if __name__ == "__main__":
    engine = GapAnalyserEngine()
    print("🔄 Running end-to-end AI career tracking pipeline...")

    # 2. Mocking intake variables for execution simulation
    user_id = "candidate_01"
    linkedin_skills = ["Backend Architecture", "PostgreSQL", "CI/CD", "Agile Methodologies"]
    repo_names = ["microservices-api-go", "stock-anomaly-detector"]
    github_skills = {"python", "flask", "docker", "machine-learning", "scikit-learn"}
    current_role = "Senior Software Engineer"

    pdf_extractor = PDFSkillExtractor()

    root_dir = Path(__file__).resolve().parent
    project_root = root_dir if (root_dir / "resources").exists() else root_dir.parent

    pdf_path = project_root / "resources" / "linkedin_profile.pdf"

    print(f"📄 Target Check: Scanning for profile document at: {pdf_path.name}")
    extracted_linkedin_skills = []

    if pdf_path.exists():
        pdf_to_text = pdf_extractor.extract_text_from_pdf(str(pdf_path))
        print(f"PDF extract {pdf_to_text}")

    # 3. Integrate & Upsert into Vector Storage
    engine.integrate_external_profile(
        user_id=user_id,
        linkedin_skills=pdf_to_text,
        github_repos=repo_names,
        github_skills=github_skills,
        current_role=current_role
    )

    # 4. Target Market Job Parameters
    target_role = "AI Engineer"
    target_job_description = (
        "We need a AI Engineer to architect high-throughput model inference pipelines. "
        "Must have deep hands-on expertise building distributed LLM applications with LangChain "
        "and managing production scaling. This role directly coordinates with enterprise product "
        "stakeholders to align tech infrastructure with client business roadmaps."
    )

    # 5. Run the Search & AI Evaluation Next Step
    print("\n🤖 Sending vectorized query profile context to GenAI Model...")
    gap_report = engine.analyze_stored_profile_gap(
        user_id=user_id,
        target_role=target_role,
        target_job_description=target_job_description
    )

    print("\n=== FINAL AI OUTPUT COMPLETED ===")
    print(gap_report)