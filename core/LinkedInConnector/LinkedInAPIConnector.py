import aiohttp
from typing import Set, Dict, Any

class LinkedInConnector:
    def __init__(self, access_token: str):
        """
        :param access_token: OAuth 2.0 User Access Token with 'r_liteprofile'
                             and 'r_member_skills' scopes.
        """
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Connection": "Keep-Alive",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        self.base_url = "https://linkedin.com"

    async def fetch_profile_and_skills(self) -> Dict[str, Any]:
        """Fetches both basic profile information and explicit member skills."""
        profile_url = f"{self.base_url}/me"
        skills_url = f"{self.base_url}/memberSkills"

        profile_data = {"name": "Unknown", "current_role": "Unknown", "skills": set()}

        async with aiohttp.ClientSession(headers=self.headers) as session:
            # 1. Fetch Profile Info (First Name, Last Name)
            async with session.get(profile_url) as p_resp:
                if p_resp.status == 200:
                    res = await p_resp.json()
                    first_name = res.get("localizedFirstName", "")
                    last_name = res.get("localizedLastName", "")
                    profile_data["name"] = f"{first_name} {last_name}".strip()

            # 2. Fetch Explicit Endorsed Skills
            async with session.get(skills_url) as s_resp:
                if s_resp.status != 200:
                    raise Exception(f"LinkedIn API Error: {s_resp.status} - {await s_resp.text()}")

                skills_payload = await s_resp.json()
                # Extract skill names from the elements collection
                elements = skills_payload.get("elements", [])

                extracted_skills = set()
                for item in elements:
                    # LinkedIn returns a reference object resolving to the skill name
                    skill_details = item.get("skill~", {})
                    if skill_details and "name" in skill_details:
                        extracted_skills.add(skill_details["name"])

                profile_data["skills"] = extracted_skills

        return profile_data
