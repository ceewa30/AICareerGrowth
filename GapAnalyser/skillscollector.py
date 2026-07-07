import sys
import os
import re
import ast
import asyncio
import httpx
from dotenv import load_dotenv
from pathlib import Path
import json

# Fixes path resolution so Python can find your 'commons' and 'Connectors' packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GitHubConnector.GitHubAPIConnector import GitHubConnector
# from LinkedInConnector.LinkedInAPIConnector import LinkedInConnector
from commons.DataValidation import UserProfile
from commons.KeywordParser import RegexKeywordParser
from commons.readme_download import download_readme, parse_readme_files, print_langchain_metrics
from commons.TargetSkill import TargetSkill
# from commons.chunk import process_and_add_repo_langchain
from commons.PDFSkillLinkedInExtractor import PDFSkillExtractor
from commons.PathCurator import DynamicPathCurator
from commons.ProfileOptimizer import PortfolioAutoOptimizer
from commons.MarketMonitor import MarketSentimentMonitor
from commons.ReportWriter import CareerRoadmapWriter
from gap_analyser_engine import GapAnalyserEngine

load_dotenv(override=True)
username = "ceewa30"
# Resolve directory paths backward to root workspace using pathlib
root_dir = Path(__file__).resolve().parent
project_root = root_dir if (root_dir / "resources").exists() else root_dir.parent
output_dir = project_root / "github_readme" / username

# def fetch_documents(output_dir, file):
#     """A homemade version of the LangChain DirectoryLoader"""

#     documents = []

#     if not output_dir.exists():
#         print(f"⚠️ Warning: Directory {output_dir} does not exist.")
#         return documents

#     # for file in output_dir.rglob("*.md"):
#     try:
#         with open(file, "r", encoding="utf-8") as f:
#             repo_name = file.stem.replace("_README", "")
#             documents.append({
#             "type": "github_readme",
#             "repo_name": repo_name,
#             "source": file.as_posix(),
#             "text": f.read()
#         })
#     except Exception as e:
#         print(f"❌ Failed to read {file.name}: {str(e)}")

#     print(f"Loaded {len(documents)} documents successfully.")
#     return documents
async def main():

    # Initialize Connectors & Core Vector DB Engine
    github_client = GitHubConnector()
    keyword_parser = RegexKeywordParser()
    engine = GapAnalyserEngine()

    repo_names = []
    master_github_skills = []
    master_linkedin_skills = []

    profile = UserProfile(
                    name=username,
                    current_role="Developer",
                    linkedin_summary="..."
                )
    final_profile = profile

    print("\n🏁 Starting Talent Acquisition Audit & Portfolio Mapping Workflow...")
    print("==========================================================")
    # Concurrently Scan GitHub Account Repos & Documents
    print(f"🔄 Auditing GitHub repository architectures for user account: '{username}'")
    # try:
    # 1. Fetch metadata skills (Languages + Topics) from GitHub API
    github_task = await github_client.fetch_skills(username=username)

    # 2. Fetch repo names to trigger README downloads
    repo_names = await github_client.fetch_repo_names(username=username)

    # 3. Download all READMEs concurrently using a single client session
    async with httpx.AsyncClient() as client:
        tasks = [github_client.fetch_readme_raw(client, username, repo) for repo in repo_names]
        results = await asyncio.gather(*tasks)

    # Filter down into a dictionary containing your actual README content strings
    captured_readmes = {r["repo"]: r["readme"] for r in results if r["readme"]}

    # 4. Save files to disk and check success metric
    file_download = download_readme(output_dir=output_dir, captured_readmes=captured_readmes)
    if file_download > 0:
        print(f"\n✅ SUCCESS: {file_download} markdown files written to disk!")
        print(f"📂 Destination Folder: {os.path.abspath(output_dir)}")

        # Physical Check: Verify files actually exist inside your workspace folder
        actual_files_on_disk = os.listdir(output_dir)
        print(f"📋 Verified Files Found in Directory ({len(actual_files_on_disk)} total):")

        final_profile = parse_readme_files(output_dir, actual_files_on_disk, profile)

        # for repo in getattr(final_profile, "github_repos", []):
        #     for chunk in getattr(repo, "readme_chunks", []):
        #         print(f"--- Chunk Index: {chunk.chunk_index} ---")
        #         print(chunk.content)
        #         print(f"Metadata: {chunk.metadata}\n")

        print_langchain_metrics(final_profile)

    else:
        print("\n❌ FAILURE or SKIPPED: No README files were saved. Check if captured_readmes was empty.")

    # 5. Parse the captured README files using Regex
    print("\n🔍 Scanning documentation markdown for hidden tools and frameworks...")
    raw_github_results = keyword_parser.parse_readmes(captured_readmes)
    parsed_github_results = raw_github_results if raw_github_results is not None else {}

    readme_skills_raw = parsed_github_results.get("flat_skills_list", [])
    readme_skills = readme_skills_raw if readme_skills_raw is not None else []

    # 🌟 SAFEGUARD 3: Absolute type guard checking for the existence and null state of github_task
    if 'github_task' in locals() and github_task is not None:
        if isinstance(github_task, list):
            clean_github_tasks = [str(t) for t in github_task if t is not None]
        else:
            clean_github_tasks = [str(github_task)]
    else:
        clean_github_tasks = []

    # 🌟 STEP B: Bulletproof Interceptor - Sanitize the dictionary data right at the source!
    clean_readme_skills = []
    for item in readme_skills:
        if isinstance(item, str):
            clean_readme_skills.append(item)
        elif hasattr(item, 'repo_name'):
            # If a RepoDocument leaked inside, extract its pure text string name
            clean_readme_skills.append(str(item.repo_name))
        else:
            continue

    seen = {}
    for skill in (clean_github_tasks + readme_skills):
        if isinstance(skill, str):
            key = skill.strip().lower()
            if key not in seen:
                seen[key] = skill
            # print(f"repo name {seen}")
        else:
            # Optional: Print a warning to identify where the bad data came from
            print(f"⚠️ Warning: Found and removed a non-string object of type {type(skill)}: {skill}")

    # Generate the clean, sorted array of strings
    master_github_skills = sorted(list(seen.values()))

    # Parse and Clean LinkedIn Profile PDF

    target_skill = TargetSkill()

    all_targets, current_role_text, verified_linkedin_skills, linkedin_summary_text = target_skill.targetskill()

    current_role = current_role_text or "Developer"
    linkedin_summary = linkedin_summary_text or ""

    # 3. Diagnostic confirmation print block
    # print("\n🚀 PIPELINE RETURN DATA RECEIVED:")
    # print(f"🔹 Returned Role string: {current_role}")
    # print(f"🔹 Verified Skills total count: {verified_linkedin_skills} & {all_targets}")
    # print(f"🔹 Summary content length: {linkedin_summary} characters")

    print(f"\n🔍 Scanning documentation markdown for hidden tools and frameworks... ")

    validated_github_skills = {s for s in master_github_skills if isinstance(s, str)}
    validated_linkedin_skills = {s for s in verified_linkedin_skills if isinstance(s, str)}

    validated_repos = []
    if hasattr(final_profile, "github_repos") and final_profile.github_repos is not None:
        validated_repos = final_profile.github_repos
    elif isinstance(final_profile, list):
        validated_repos = final_profile
    else:
        validated_repos = []


    # 6. Instantiate your Pydantic model
    user_profile = UserProfile(
        name=username,
        current_role=current_role,
        github_skills=validated_github_skills,
        github_repos=validated_repos,
        linkedin_skills=validated_linkedin_skills,
        linkedin_summary=linkedin_summary
    )

    # Upsert flattened textual vectors into local ChromaDB memory instance
    engine.integrate_external_profile(
        user_id=user_profile.name,
        linkedin_skills=user_profile.linkedin_skills if user_profile.linkedin_skills is not None else set(),
        linkedin_summary=user_profile.linkedin_summary or "",
        github_repos=user_profile.github_repos,  # Safe clean string names array
        github_skills=user_profile.github_skills if user_profile.github_skills is not None else set(),
        current_role=user_profile.current_role or "Developer"
    )

    # Run Semantic Gap Analysis for a Target Role
    # Options from your market_standards.yaml benchmarks: "Software Engineer", "Senior Software Engineer", "Tech Lead"
    target_role_query = "AI Engineer"
    target_job_description = (
    "We need a AI Engineer to architect high-throughput model inference pipelines. "
    "Must have deep hands-on expertise building distributed LLM applications with LangChain "
    "and managing production scaling. This role directly coordinates with enterprise product "
    "stakeholders to align tech infrastructure with client business roadmaps."
)
    analysis_report = engine.analyze_stored_profile_gap(user_id=user_profile.name, target_role=target_role_query, target_job_description=target_job_description)
    # analysis_report = engine.generate_gap_analysis(user_id=user_profile.name, target_role=target_role_query)

    # 10. Display Actionable Career Roadmap Metrics
    print("\n📊 ================= GAP ANALYSIS REPORT =================")
    if "error" in analysis_report:
        print(f"❌ Analysis Aborted: {analysis_report['error']}")
    else:
        print(analysis_report)
    print("==========================================================")

    # Inject Dynamic Learning Path Curation parameters
    curator = DynamicPathCurator()

    user_preference_style = "Theoretical" # Options: Visual, Practical, Theoretical
    user_time_window = "Long"            # Options: Short, Long

    missing_gaps_list = analysis_report.get("missing_skills_gap", [])

    curated_learning_path = curator.curate_path(
        missing_gaps=missing_gaps_list,
        learning_style=user_preference_style,
        time_availability=user_time_window
    )

    print("\n📊 ================= GAP ANALYSIS REPORT =================")
    # ... (Print candidate names and missing skills exactly as your stable script does)

    print(f"\n🎓 ====== DYNAMIC LEARNING PATH (Preferences: {user_preference_style} / {user_time_window}) ======")
    if not curated_learning_path:
        print("  🎉 No learning gaps found, or catalog options do not match active constraints.")
    else:
        for skill_gap, recommendations in curated_learning_path.items():
            print(f"\n  🎯 Targeted Skill Gap Development: {skill_gap}")
            for rec in recommendations:
                print(f"    🔹 [{rec['type']}] {rec['title']}")
                print(f"       Accuracy: {rec['match_accuracy']} | Resource Access: {rec['url']}")
    print("==========================================================")

    optimizer = PortfolioAutoOptimizer(db_engine=engine)

    print("\n⚡ [Event Loop] User completed 'System Design Fundamentals' from learning path...")

    # 1. Update the Pydantic data layers and Vector database concurrently in memory
    optimized_profile = optimizer.acquire_new_skill(
        user_profile,
        "System Design",
        repo_names
    )

    # 2. Compile recruiter algorithm friendly resume documentation markdown blocks
    ats_cv_data = optimizer.generate_ats_optimized_cv_block(optimized_profile)

    # 3. Print out the optimized layout summary metrics
    print("\n🚀 ====== ATS RECRUITER KEYWORD EXTRACTION PREVIEW ======")
    print(ats_cv_data)
    print("==========================================================")

    # 4. Re-run Gap Analysis to instantly verify the skill gap is closed
    updated_analysis = engine.generate_gap_analysis(user_id=optimized_profile.name, target_role="Senior Software Engineer")
    print(f"🚨 Updated Growth Gaps (MISSING SKILLS): {updated_analysis['missing_skills_gap'] if updated_analysis['missing_skills_gap'] else 'None (All Closed!)'}")

    # Regional Market Sentiment & Compensation Audit
    monitor = MarketSentimentMonitor()

    target_geo_market = "US-East"
    sentiment_report = monitor.track_regional_value(optimized_profile, target_geo_market)

    print(f"\n📈 ====== MARKET SENTIMENT EVALUATION (Region: {target_geo_market}) ======")
    if "error" in sentiment_report:
        print(f"❌ Monitor Alert: {sentiment_report['error']}")
    else:
        print(f"💵 Regional Base Median : {sentiment_report['base_market_median']} {sentiment_report['currency']}")
        print(f"📊 Estimated Value      : {sentiment_report['estimated_market_value']} {sentiment_report['currency']} ({sentiment_report['premium_multiplier_applied']} Skill Premium)")
        print(f"🔥 Highly Leveraged Assets: {', '.join(sentiment_report['high_value_assets']) if sentiment_report['high_value_assets'] else 'None'}")
        print(f"🔔 System Alert Status  : {sentiment_report['alert_message']}")
    print("==========================================================")

    # Setup resource destination targets using pathlib attributes already defined at the top
    resources_dir = project_root / "resources"
    report_writer = CareerRoadmapWriter(output_dir=resources_dir)

    print("\n💾 Exporting consolidated architecture analytics to disk...")
    file_io_result = report_writer.write_report(
        profile=optimized_profile,               # Uses the real-time optimized profile state
        analysis_report=updated_analysis,        # Uses the post-optimized gap assessment metadata
        curated_learning_path=curated_learning_path,
        sentiment_report=sentiment_report,
        ats_cv_data=ats_cv_data
    )
    print(file_io_result)
    print("==========================================================\n")


    # except Exception as e:
    #     print(f"❌ Failed to extract profile data: {e}")

if __name__ == "__main__":
    asyncio.run(main())
