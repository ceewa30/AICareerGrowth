import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from commons.PDFSkillLinkedInExtractor import PDFSkillExtractor
from pathlib import Path
import ast
import re

root_dir = Path(__file__).resolve().parent
project_root = root_dir if (root_dir / "resources").exists() else root_dir.parent

class TargetSkill:
    def __init__(self):
        self.pdf_extractor = PDFSkillExtractor()
        self.pdf_path = project_root / "resources" / "linkedin_profile.pdf"
        self.text_skill = project_root / "resources" / "target_skills.txt"

    def targetskill(self):
        print(f"📄 Target Check: Scanning for profile document at: {self.pdf_path.name}")
        extracted_linkedin_skills = []

        if self.pdf_path.exists():
            pdf_to_text = self.pdf_extractor.extract_text_from_pdf(str(self.pdf_path))
            parsed_profile = self.pdf_extractor.parse_skills_section(pdf_to_text)

            if parsed_profile is None:
                parsed_profile = {}

            combined_profile_text = " ".join(parsed_profile.values())

            with open(self.text_skill, "r", encoding="utf-8") as f:
                raw_content = f.read().strip()

            try:
                # 1. Clean the string to isolate the actual array assignment right side block
                if "target_skills =" in raw_content:
                    array_string = raw_content.split("target_skills =", 1)[1].strip()
                else:
                    array_string = raw_content

                # 2. Convert the code-like text block string into a genuine Python list array object
                target_skills = ast.literal_eval(array_string)
            except Exception as e:
                print(f"⚠️ parsing failed due to file structure, falling back to clean splitlines strategy: {e}")
                # Secondary fallback layout cleaner
                target_skills = [
                    skill.replace('"', '').replace("'", "").replace(",", "").replace("[", "").replace("]", "").strip()
                    for line in raw_content.splitlines()
                    for skill in line.split()
                    if skill.strip()
                ]

            # 3. Filter out the literal assignment text fragments if any slipped through
            target_skills = [s for s in target_skills if "target_skills" not in s and "=" not in s]

            for skill in target_skills:
                pattern = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
                if pattern.search(combined_profile_text):
                    extracted_linkedin_skills.append(skill)

            verified_linkedin_skills = [s for s in extracted_linkedin_skills if len(str(s).strip()) > 1]

            print("\n✅ Extraction successful! Found sections:")
            for section, content in parsed_profile.items():
                print(f"🔹 {section} ({len(content.splitlines())} lines of data)")

            linkedin_skills_text = parsed_profile.get("Top Skills", "")
            words_to_remove = re.compile(r"\b(Docker|PostgreSQL|SQLite)\b", re.IGNORECASE)

            # Substitute the matched words with an empty string
            profile_summary = words_to_remove.sub("", linkedin_skills_text)
            # Clean up any accidental double spaces created by removing the words
            profile_summary = re.sub(r"\s+", " ", profile_summary).strip()

            header_block = parsed_profile.get("Contact/Header Info", "")
            clean_text = str(header_block).replace('\\n', ' ').replace('\n', ' ')

            title_pattern = re.search(r"07045\s+(.*?)\s+\d{10}", clean_text)
            only_title = title_pattern.group(1).strip() if title_pattern else "Title Not Found"
            current_role = only_title.split("|")[0].strip()

            # Check the native dictionary key first
            raw_summary_block = parsed_profile.get("Summary", "").strip()
            if raw_summary_block:
                only_summary = re.sub(r'\s+', ' ', raw_summary_block).rstrip("']")
            else:
                summary_pattern = re.search(r"\(Portfolio\)\s+(.*?)(?=\s*Experience|\Z)", clean_text, re.IGNORECASE)
                if summary_pattern:
                    only_summary = re.sub(r'\s+', ' ', summary_pattern.group(1).strip()).rstrip("']")
                else:
                    only_summary = "Summary Not Found"

            linkedin_summary = only_summary + profile_summary.strip()

            # --- Output Result Pipeline ---
            # print("\n=========================================================================")
            # print(f"👤 Current Role: {current_role}")
            # print(f"🛠️ LinkedIn Skills: {verified_linkedin_skills}")
            # print(f"📄 Profile Summary: {linkedin_summary}")
            # print("=========================================================================")
            self.linkedin_payload = {
                "current_role": current_role,
                "linkedin_skills": verified_linkedin_skills,
                "linkedin_text": pdf_to_text,
                "linkedin_summary": linkedin_summary,
                "target_skills": target_skills
                }
            return self.linkedin_payload

        else:
            print("⚠️ 'linkedin_profile.pdf' not found! Falling back to standard mock arrays.")

            # # 🌟 FIXED: Define clean mock variables for your fallback path
            # target_skills = ["Python", "PHP", "MySQL", "JavaScript", "HTML", "CSS", "AWS", "Docker"]
            # current_role = "Full Stack Developer"
            # verified_linkedin_skills = ["Python", "PHP", "MySQL", "JavaScript", "HTML", "CSS"]
            # linkedin_summary = "Certified Associate in Python Programming with expertise in web application design."

            # # 🌟 FIXED: Consistently return all 4 items so unpacking never crashes!
            # return target_skills, current_role, verified_linkedin_skills, linkedin_summary



if __name__ == "__main__":
    target_skill = TargetSkill()

    linkedin_data = target_skill.targetskill()

    # 3. Diagnostic confirmation print block
    print("\n🚀 PIPELINE RETURN DATA RECEIVED:")
    print(f"🔹 Returned Role string: {linkedin_data['current_role']}")
    print(f"🔹 Verified Skills total count: {linkedin_data['linkedin_skills']}")
    print(f"🔹 Summary content length: {linkedin_data['linkedin_summary']} characters")
