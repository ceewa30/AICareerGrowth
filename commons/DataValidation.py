from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field, ConfigDict, computed_field

# =====================================================================
# 1. AI COACHING & AGENT GENERATION OUTCOMES
# =====================================================================
class PathwayStep(BaseModel):
    """A single sequential course milestone within the upskilling path."""
    certification_or_course: str = Field(description="Exact, verified name of the educational credential.")
    provider: str = Field(description="The authoritative institution issuing the track.")
    focus_skills_addressed: List[str] = Field(description="The specific deep-tech capabilities gained.")
    duration_weeks: int = Field(description="Estimated weekly commitment to finish.")
    strategic_rationale: str = Field(description="Why this is critical for a senior engineer making this pivot based on their combined background.")

class UnifiedCareerReport(BaseModel):
    """The master target payload format required by OpenAI's structured parse completion endpoint."""
    current_strengths_detected: List[str] = Field(description="Core engineering capabilities derived from both Resume and GitHub.")
    core_gaps_identified: List[str] = Field(description="Underlying core knowledge gaps to fix to meet the target goal.")
    curated_roadmap: List[PathwayStep] = Field(description="Sequential, non-redundant educational steps.")


# =====================================================================
# 2. CHROMADB METADATA STORAGE COMPLIANCE SCHEMA
# =====================================================================
class DynamicRepoMetadata(BaseModel):
    """Clean, flat key structure for ChromaDB metadata injection."""
    author: str = Field(description="The primary developer or contributor name found in the file.")

    tech_stack: str = Field(description="A comprehensive inventory of all core technical proficiencies, including programming languages, operating system, repository (e.g., Git, GitHub), frameworks, " \
    "cloud platforms, architecture patterns (e.g., Microservices, REST APIs), databases (e.g., SQL, MySQL, PostSQL), and developer methodologies explicitly mentioned in the text.")

    uses_machine_learning: bool = Field(description="True if machine learning models (e.g., classification, regression, NLP, or neural networks) are utilized.")

    algorithm_summary: str = Field(description="Short phrase explaining the specific algorithmic approach used.")
    current_role: str = Field(description="The current job title or professional headline")

class ResumeExtractionSchema(BaseModel):
    """Temporary structured output target for raw resume parsing."""
    name: str = Field(description="The user's extracted full name")
    current_role: str = Field(description="The current job title or professional headline")
    extracted_skills: List[str] = Field(description="List of core technical skills explicitly found in the text")
    professional_summary: str = Field(description="A concise professional summary extracted from the text")

# =====================================================================
# 3. VECTOR INFRASTRUCTURE DATA CAPTURE LAYER
# =====================================================================
class READMEChunk(BaseModel):
    """Stores a single optimized text block from a repository README file."""
    chunk_id: str = Field(description="Unique deterministic ID for the vector store row")
    chunk_index: int = Field(description="The sequential position index of the chunk")
    content: str = Field(description="The text or code content inside this specific chunk")
    embedding: List[float] = Field(default_factory=list, description="1536-dim OpenAI vector array")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context mapping tags")


class RepoDocument(BaseModel):
    """Represents a repository name alongside its README documentation assets."""
    repo_name: str = Field(description="The short name of the GitHub repository.")
    raw_readme: Optional[str] = Field(default=None, description="The full raw Markdown text string from the README.md file.")
    readme_chunks: List[READMEChunk] = Field(default_factory=list, description="The README split down into structured chunks optimized for LLM consumption.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


# =====================================================================
# 4. UNIFIED DYNAMIC CANDIDATE CONTAINER
# =====================================================================
class UserProfile(BaseModel):
    """Aggregates all user metadata, verified repository assets, and extracted skills."""
    name: str = Field(description="The user's legal or profile name.")
    current_role: str = Field(description="The candidate's current professional job title.")
    github_skills: List[str] = Field(default_factory=list, description="Programming languages and framework tags extracted from GitHub.")
    linkedin_skills: List[str] = Field(default_factory=list, description="Skills explicitly extracted from Resume or LinkedIn parsing loops.")
    linkedin_summary: str = Field(default="No profile summary text payload.", description="Full text bio summary block.")

    @computed_field
    @property
    def all_skills(self) -> Set[str]:
        """Combines and normalizes candidate skills to lowercase for instant mathematical sets logic."""
        combined = set(self.github_skills).union(set(self.linkedin_skills))
        return {str(s).strip().lower() for s in combined if s}


class TargetRoleRequirements(BaseModel):
    """Industry standard benchmark baseline for target tracking paths."""
    title: str = Field(description="The targeted aspirational job role title.")

    # REFACTOR: Changed to List to ensure compatibility with frontend JSON data parsers
    required_technical_skills: List[str] = Field(default_factory=list, description="Hard computational and architectural engineering expectations.")
    required_leadership_skills: List[str] = Field(default_factory=list, description="Management, mentorship, and operational leadership criteria.")

    @computed_field
    @property
    def all_required(self) -> Set[str]:
        """Combines and normalizes all required benchmark skills to lowercase."""
        combined = set(self.required_technical_skills).union(set(self.required_leadership_skills))
        return {str(s).strip().lower() for s in combined if s}


class GapAnalysisReport(BaseModel):
    user_name: str
    target_role: str
    matching_skills: List[str]
    missing_technical_skills: List[str]
    missing_leadership_skills: List[str]
    ready_percentage: float


class ResumeExtractionSchema(BaseModel):
    """Temporary structured output target for raw resume parsing."""
    name: str = Field(description="The user's extracted full name")
    current_role: str = Field(description="The current job title or professional headline")
    extracted_skills: List[str] = Field(description="List of core technical skills explicitly found in the text")
    professional_summary: str = Field(description="A concise professional summary extracted from the text")