import re
from typing import Tuple, Optional

class IngestionGuardrails:
    def __init__(self, max_characters: int = 40000):
        """
        Configures operational parameters and security thresholds for input files.
        Default limit (~40k chars) allows robust multi-page profiles while blocking overflows.
        """
        self.max_characters = max_characters

        # Comprehensive regex array tracking common LLM prompt override patterns
        self.malicious_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s*prompt\s*override",
            r"you\s+must\s+now\s+output",
            r"forget\s+(your\s+)?core\s+objective",
            r"readiness\s*score\s*:\s*100%",
            r"candidate\s+is\s+perfect"
        ]

    def sanitize_raw_text(self, raw_text: Optional[str]) -> str:
        """
        Strips away system control symbols, excessive white spaces,
        and hidden binary characters that break downstream chunk tokens.
        """
        if not raw_text:
            return ""

        # 1. Remove non-printable control characters, preserving basic newlines/tabs
        cleaned = "".join(ch for ch in raw_text if ch.isprintable() or ch in "\n\r\t")

        # 2. Flatten massive line spacing stacks to prevent artificial context inflation
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)

        return cleaned.strip()

    def evaluate_input_safety(self, raw_content: str) -> Tuple[bool, str]:
        """
        Executes structural audit verification checks over incoming raw profile texts.
        Returns a tuple: (is_safe: bool, resolution_message: str)
        """
        # Check 1: Enforce physical text boundary limits
        if len(raw_content) > self.max_characters:
            error_msg = f"Security Reject: Profile length ({len(raw_content)} chars) exceeds allowed limit of {self.max_characters}."
            print(f"❌ {error_msg}")
            return False, error_msg

        # Check 2: Pre-sanitize whitespace to expose obscured injection text chunks
        normalized_text = self.sanitize_raw_text(raw_content)
        normalized_lower = normalized_text.lower()

        # Check 3: Screen text parameters for malicious instructions or score hijacking
        for pattern in self.malicious_patterns:
            if re.search(pattern, normalized_lower):
                security_msg = "Security Alert: Malicious instructions or prompt override attempts detected in source text."
                print(f"🚨 [Guardrails Audit] Rejecting input file payload. Pattern matched: '{pattern}'")
                return False, security_msg

        # Check 4: Block completely blank or unparseable input states
        if len(normalized_text) < 50:
            validation_msg = "Validation Error: Provided text profile content is too short to run an accurate career mapping."
            return False, validation_msg

        return True, "Success: Text segment passed all safety parameters."
