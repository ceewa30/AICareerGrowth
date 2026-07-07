import re

class RegexKeywordParser:
    def __init__(self):
        """A dictionary mapping common technology domains to lists of keywords we want to track"""
        self.skill_keywords = {
            "Frameworks & Libraries": [
                "FastAPI", "Django", "Flask", "React", "Next.js", "Vue", "Angular",
                "Spring Boot", "Express", "Laravel", "Tailwind", "Bootstrap", "Pandas", "NumPy"
            ],
            "DevOps & Infrastructure": [
                "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Ansible",
                "CI/CD", "GitHub Actions", "Jenkins", "Linux", "Nginx"
            ],
            "Databases & Caching": [
                "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "DynamoDB", "Elasticsearch"
            ],
            "Architecture & Methodologies": [
                "REST API", "GraphQL", "Microservices", "gRPC", "WebSockets", "Agile", "Scrum"
            ]
        }

    def parse_readmes(self, captured_readmes: dict) -> dict:
        """
        Scans a dictionary of {repo_name: readme_text} and returns detected
        keywords grouped by domain, along with a flat unique list.
        """
        detected_by_category = {category: set() for category in self.skill_keywords}
        all_detected_skills = set()

        # Combine all documentation text into a single block for fast scanning
        combined_text = " ".join([text for text in captured_readmes.values() if text])

        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                pattern = rf"\b{re.escape(keyword)}\b"

                if re.search(pattern, combined_text, re.IGNORECASE):
                    detected_by_category[category].add(keyword)
                    all_detected_skills.add(keyword)

        return {
            "categorized_skills": {cat:sorted(list(skills)) for cat, skills in detected_by_category.items()},
            "flat_skills_list": sorted(list(all_detected_skills))
        }

    def parse_linkedin_skills(self, extracted_linkedin_skills: list) -> dict:
        """
        Scans a list of skills and returns detected keywords grouped by domain along with a flat unique list.
        """
        # 1. Initialize empty tracking sets for each distinct category key
        detected_by_category = {category: set() for category in self.skill_keywords}
        all_detected_skills = set()

        # 2. Combine the individual list tokens into one flat searchable text string
        combined_text = " ".join([text for text in extracted_linkedin_skills if text])

        # 3. Double-loop to map over both the categories and their nested list arrays
        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                # Use word boundaries (\b) to prevent incorrect substring matches
                pattern = rf"\b{re.escape(keyword)}\b"

                if re.search(pattern, combined_text, re.IGNORECASE):
                    detected_by_category[category].add(keyword)
                    all_detected_skills.add(keyword)

        # 4. Return clean, sorted standard arrays inside a structured dictionary
        return {
            "categorized_skills": {cat: sorted(list(skills)) for cat, skills in detected_by_category.items()},
            "flat_skills_list": sorted(list(all_detected_skills))
        }
