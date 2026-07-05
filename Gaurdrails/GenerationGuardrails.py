from typing import Dict, Any, List, Optional
import json

class GenerationGuardrails:
    def __init__(self, mandatory_certification_providers: Optional[List[str]] = None, maximum_skills_allowed: int = 10):
        """
        Configures the generation validation gateway thresholds.
        Ensures recommendations align with established training channels.
        """
        self.approved_providers = mandatory_certification_providers or [
            "AWS", "Google Cloud", "Azure", "Coursera", "DeepLearning.AI",
            "CompTIA", "Kubernetes", "PostgreSQL", "HashiCorp"
        ]
        self.max_limit = maximum_skills_allowed

    def _sanitize_token(self, token: str) -> str:
        """
        Cleans messy scraped artifacts, text dumps, or parentheses leaks
        (e.g., converts 'Athena)' to 'athena', or 'AWS (EMR' to 'aws emr').
        """
        if not token:
            return ""
        cleaned = token.lower().strip()
        cleaned = cleaned.replace(")", "").replace("(", "")
        return " ".join(cleaned.split())

    def _is_provider_valid(self, text: str) -> bool:
        """
        Checks if the provided text mentions any of the approved certification/training providers.
        """
        if not text:
            return False
        text_lower = text.lower()
        return any(provider.lower() in text_lower for provider in self.approved_providers)

    def verify_and_align_output(
        self,
        engine_readiness_score: float,
        engine_missing_gaps: List[str],
        raw_llm_payload: Any
    ) -> Dict[str, Any]:
        """
        Validates the output of your LLM against native backend calculations.
        Evicts tool hallucinations, enforces limits, and structures output for the UI.
        """
        print("🛡️ [Generation Guardrails] Commencing post-generation output consistency check...")

        # 1. Fallback Trap: Handle cases where the OpenAI call dropped out or returned None
        if raw_llm_payload is None:
            print(" -> Guardrail Alert: OpenAI payload missing. Instantiating schema safety fallback state.")
            return self._generate_safe_fallback(engine_readiness_score, engine_missing_gaps)

        try:
            # 2. Extract internal fields dynamically regardless of dictionary or object formats
            is_dict = isinstance(raw_llm_payload, dict)

            if is_dict:
                llm_strengths = raw_llm_payload.get('current_strengths_detected', raw_llm_payload.get('existing_strengths', []))
                roadmap_source = raw_llm_payload.get("phased_upskilling_roadmap", [])
            else:
                llm_strengths = getattr(raw_llm_payload, 'current_strengths_detected', getattr(raw_llm_payload, 'existing_strengths', []))
                roadmap_source = getattr(raw_llm_payload, "phased_upskilling_roadmap", [])

            # 3. Generate robust set of sanitized, authorized ground truth baseline skill tokens
            authorized_tokens = set()
            for skill in engine_missing_gaps:
                sanitized = self._sanitize_token(skill)
                if sanitized:
                    authorized_tokens.add(sanitized)

            print(" -> Reconciling generated roadmap steps against ground truths...")
            sanitized_roadmap_steps = []

            # 4. Iterate through generated steps to evict hallucinations and enforce limits
            for step in roadmap_source:
                step_dict = step if isinstance(step, dict) else getattr(step, '__dict__', {})
                step_tool = step_dict.get("target_tool", "") if isinstance(step, dict) else getattr(step, "target_tool", "")
                step_desc = step_dict.get("description", "") if isinstance(step, dict) else getattr(step, "description", "")

                if not step_tool:
                    continue

                sanitized_tool = self._sanitize_token(step_tool)

                # GUARDRAIL A: Evict tool hallucinations not explicitly found in missing gaps vector
                if sanitized_tool not in authorized_tokens:
                    print(f" ⚠️ Guardrails Eviction: Suggested tool '{step_tool}' is not an authorized gap.")
                    continue

                # Optional Quality Check: Inject approved providers if description lacks clear channels
                if not self._is_provider_valid(step_desc):
                    fallback_provider = "Coursera or standard documentation channels"
                    if "aws" in sanitized_tool: fallback_provider = "AWS Certification Paths"
                    elif "kubernetes" in sanitized_tool: fallback_provider = "CNCF Kubernetes Pathways"

                    if isinstance(step, dict):
                        step["description"] = f"{step_desc} Recommended study channel: {fallback_provider}.".strip()
                    else:
                        setattr(step, "description", f"{step_desc} Recommended study channel: {fallback_provider}.".strip())

                sanitized_roadmap_steps.append(step if isinstance(step, dict) else step_dict)

                # GUARDRAIL B: Enforce explicit max cap limit boundary (Top 10)
                if len(sanitized_roadmap_steps) >= self.max_limit:
                    break

            # 5. Handle Depth Remediation Strategy: If roadmap items are missing, generate baseline text block
            if len(sanitized_roadmap_steps) == 0:
                print(" -> Guardrail Triggered: Generated roadmap lacks verified items. Injecting baseline text.")
                llm_roadmap_text = self._compile_structured_baseline_roadmap(engine_missing_gaps)
            else:
                # Convert structured validated steps list into clean displayable Markdown text
                llm_roadmap_text = "### 🎯 Recommended Sequential Upskilling Roadmap (Verified Top 10)\n\n"
                for idx, step in enumerate(sanitized_roadmap_steps):
                    phase = 1 if idx < 3 else (2 if idx < 7 else 3)
                    tool_name = step.get("target_tool", "Unknown Tool")
                    desc_text = step.get("description", "Complete micro-project implementation exercises.")
                    llm_roadmap_text += f"#### Phase {phase} - Step {idx + 1}: Master {tool_name}\n"
                    llm_roadmap_text += f"* **Focus Item**: {desc_text}\n\n"

            # 6. Pack final payload parameters into a consistent output container state
            sanitized_output = {
                "readiness_score": float(engine_readiness_score),  # Enforce exact engine numerical truth
                "current_strengths_detected": list(llm_strengths),
                "phased_roadmap_text": llm_roadmap_text,
                "mathematically_verified_gaps": list(engine_missing_gaps)[:self.max_limit]
            }

            print(f"✅ [Generation Guardrails] Output verified. Dataset locked at {len(sanitized_roadmap_steps)} verified path steps.")
            return sanitized_output

        except Exception as e:
            print(f"❌ Guardrail Execution Fault: Failed to compile consistency checks: {str(e)}")
            return self._generate_safe_fallback(engine_readiness_score, engine_missing_gaps)

    def _compile_structured_baseline_roadmap(self, missing_gaps: List[str]) -> str:
        """
        Programmatically builds a fallback Markdown roadmap if the LLM output is malformed.
        """
        markdown = "### 🚀 Personalized Upskilling Action Plan (System Generated Baseline)\n\n"
        markdown += "Your profile is missing key technical competencies required for this target role. Follow this structured upskilling path:\n\n"

        # Limit fallback generator to max allowed items to avoid visual truncation issues
        for idx, gap in enumerate(missing_gaps[:self.max_limit]):
            phase = 1 if idx < 3 else (2 if idx < 7 else 3)
            markdown += f"#### Phase {phase} - Step {idx + 1}: Remediate {gap.upper()}\n"
            markdown += f"* **Action Item**: Complete specialized training for **{gap}** using official provider tracks or Coursera.\n"
            markdown += f"* **Practical Exercise**: Build a standalone GitHub repository implementing a micro-project focused on {gap}.\n\n"
        return markdown

    def _generate_safe_fallback(self, score: float, gaps: List[str]) -> Dict[str, Any]:
        """
        Generates a non-crashing structural object layout if the API connection fails entirely.
        """
        return {
            "readiness_score": score,
            "current_strengths_detected": ["Context analysis complete. See roadmap for details."],
            "phased_roadmap_text": self._compile_structured_baseline_roadmap(gaps),
            "mathematically_verified_gaps": gaps[:self.max_limit]
        }
