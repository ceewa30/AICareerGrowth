import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from commons.DataValidation import TargetRoleRequirements, UserProfile, GapAnalysisReport

# =====================================================================
# 2. THE ERROR-FREE GAP ANALYSIS ENGINE
# =====================================================================
class GapAnalysisEngine:
    @staticmethod
    def analyze(user: UserProfile, target: TargetRoleRequirements) -> GapAnalysisReport:
        # Extract fully normalized lowercase string sets
        user_skills = user.all_skills
        target_skills = target.all_required

        # FIX 1: Overlapping math execution must intersect lowercase strings to find real matches
        matching_skills = user_skills.intersection(target_skills)

        # FIX 2: Check lowercase properties but preserve clear formatting for the UI/Terminal view layers
        missing_tech = [
            skill for skill in target.required_technical_skills
            if skill.strip().lower() not in user_skills
        ]

        missing_leadership = [
            skill for skill in target.required_leadership_skills
            if skill.strip().lower() not in user_skills
        ]

        # Calculate overall readiness score based on normalized total targets
        total_required = len(target_skills)
        ready_percentage = (len(matching_skills) / total_required * 100) if total_required > 0 else 100.0

        return GapAnalysisReport(
            user_name=user.name,
            target_role=target.title,
            matching_skills=sorted(list(matching_skills)),
            missing_technical_skills=sorted(missing_tech),
            missing_leadership_skills=sorted(missing_leadership),
            ready_percentage=round(ready_percentage, 2)
        )

# =====================================================================
# 3. VERIFIED DATA PROCESSING LIFECYCLE RUN
# =====================================================================
if __name__ == "__main__":
    # FIX 3: Added missing mandatory schema fields to clear Pydantic initialization checks
    mock_user = UserProfile(
        name="Alex Mercer",
        current_role="Senior Developer",
        github_skills={"Python", "Go", "Docker", "Kubernetes", "PostgreSQL"},
        linkedin_skills={"Python", "Backend Development", "Agile Methodologies", "Code Review"},
        linkedin_summary="Experienced systems engineer focused on high-throughput backend services."
    )

    target_lead_role = TargetRoleRequirements(
        title="Lead Developer",
        required_technical_skills={"Python", "Go", "System Design", "Cloud Architecture", "Kubernetes"},
        required_leadership_skills={"Stakeholder Management", "Mentorship", "Agile Methodologies", "Resource Planning"}
    )

    # Run calculation pipeline
    report = GapAnalysisEngine.analyze(mock_user, target_lead_role)

    print(f"=== GAP ANALYSIS REPORT FOR {report.user_name.upper()} ===")
    print(f"Target Goal: Move to {report.target_role}")
    print(f"Current Role Readiness: {report.ready_percentage}%\n")

    print("✅ Skills You Have (Normalized):")
    for skill in report.matching_skills:
        print(f"  - {skill.title()}")

    print("\n🚨 Missing Technical Competencies:")
    if report.missing_technical_skills:
        for skill in report.missing_technical_skills:
            print(f"  - {skill}")
    else:
        print("  - None! Technical skills match perfectly.")

    print("\n🚨 Missing Leadership/Management Competencies:")
    if report.missing_leadership_skills:
        for skill in report.missing_leadership_skills:
            print(f"  - {skill}")
    else:
        print("  - None! Leadership skills match perfectly.")
