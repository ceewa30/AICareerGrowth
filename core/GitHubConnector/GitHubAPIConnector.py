import httpx
import asyncio
import os

class GitHubConnector:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN_KEY')
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}" if self.token else ""

    async def fetch_repo_names(self, username: str) -> list:
        """Fetches a list of all public repository names for the user."""
        url = f"{self.base_url}/users/{username}/repos"
        print(f"📡 Making API Request to: {url}")
        params = {"per_page": 100, "sort": "updated"}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            if response.status_code == 404:
                raise Exception(f"GitHub API Error: 404 - User '{username}' Not Found")
            elif response.status_code != 200:
                raise Exception(f"GitHub API Error: {response.status_code} - {response.text}")

            repos = response.json()
            clean_repo_names = []
            for repo in repos:
                if repo.get("fork", False):
                    continue

                owner_data = repo.get("owner", {})
                if owner_data.get("login", "").lower() != username.lower():
                    continue

                clean_repo_names.append(repo["name"])

            return clean_repo_names

    async def fetch_skills(self, username: str) -> list:
        """Fetches repositories and merges languages and topics into a single flat list of skills"""
        url = f"{self.base_url}/users/{username}/repos"
        print(f"📡 Making API Request to: {url}")
        params = {"per_page": 100, "sort": "updated"}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            if response.status_code == 404:
                raise Exception(f"GitHub API Error: 404 - User '{username}' Not Found")
            elif response.status_code != 200:
                raise Exception(f"GitHub API Error: {response.status_code} - {response.text}")

            repos = response.json()
            # print(f"Repos: {repos}")
            skills = set()
            for repo in repos:
                # Skip fork repository completely
                if repo.get("fork", False):
                    continue

                owner_data = repo.get("owner", {})
                if owner_data.get("login", "").lower() != username.lower():
                    continue
                # 1. Extract automated core language
                language = repo.get("language")
                if language:
                    skills.add(language)

                # 2. Extract manual topic tags safely
                repo_topics = repo.get("topics", [])
                for topic in repo_topics:
                    skills.add(topic)

            return list(skills)

    async def fetch_readme_raw(self, client: httpx.AsyncClient, username: str, repo: str) -> dict:
        """Fetches the raw text of a specific repository's README safely."""
        url = f"{self.base_url}/repos/{username}/{repo}/readme"
        headers = self.headers.copy()
        headers["Accept"] = "application/vnd.github.raw+json"

        try:
            response = await client.get(
                url,
                headers=headers,
                timeout=5.0,
                follow_redirects=True
                )

            if response.status_code == 200:
                return {"repo": repo, "readme": response.text}
            return {"repo": repo, "readme": None}
        except Exception:
            return {"repo": repo, "readme": None}


async def main():
    github_client = GitHubConnector()
    username = "ceewa30"

    # 1. Fetch skills (this hits the API once)
    github_skills = await github_client.fetch_skills(username=username)

    # 2. Fetch repo names directly to trigger README downloads
    repo_names = await github_client.fetch_repo_names(username=username)

    # 3. Download all READMEs concurrently using a single client session
    async with httpx.AsyncClient() as client:
        tasks = [github_client.fetch_readme_raw(client, username, repo) for repo in repo_names]
        results = await asyncio.gather(*tasks)

        # Filter down into a dictionary containing your actual README content strings
        captured_readmes = {r["repo"]: r["readme"] for r in results if r["readme"]}

    print(f"✅ Extracted {len(github_skills)} global skills!")
    print(f"📖 Downloaded {len(captured_readmes)} raw markdown README files!")


if __name__ == "__main__":
    asyncio.run(main())
