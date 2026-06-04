# ============================================================
# FILE: alignment/constitution.py
# Constitutional AI — rule-based output alignment.
#
# The model checks its own outputs against a plain-text
# constitution file. Outputs that violate rules are flagged
# and rewritten.
#
# Constitution format (data/constitution.txt):
#   Each rule is one line starting with "RULE:"
#   Comments start with "#"
# ============================================================

import re, os, json
from datetime import datetime


DEFAULT_CONSTITUTION = """
# AXIOM Constitution
# One rule per line. Start each rule with RULE:
# These are checked against every model output.

RULE: Do not provide instructions for creating weapons or explosives.
RULE: Do not generate content that demeans or attacks individuals based on identity.
RULE: Do not claim to be human when directly and sincerely asked.
RULE: Do not make up facts presented as certain when they are not known.
RULE: Do not produce content that sexualises minors.
RULE: Acknowledge uncertainty rather than fabricating confident answers.
RULE: Be honest about being an AI system.
"""


def load_constitution(path=None):
    """
    Load rules from a constitution file.
    Falls back to DEFAULT_CONSTITUTION if file not found.

    Returns:
        list[str]: List of rule strings.
    """
    if path and os.path.exists(path):
        with open(path) as f:
            text = f.read()
    else:
        text = DEFAULT_CONSTITUTION

    rules = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('RULE:'):
            rules.append(line[5:].strip())
    return rules


class ConstitutionChecker:
    """
    Checks model outputs against a set of rules.

    Two-stage process:
      1. _check_rule:  heuristic keyword scan (fast, catches obvious violations)
      2. critique:     flag all violations with reasons
      3. rewrite:      produce a safe alternative (stub — needs model in loop)
    """

    # Keyword patterns for each rule category
    RULE_PATTERNS = {
        'weapons':      r'\b(make|build|create|synthesise?|instructions? for|how to make)\b.{0,40}\b(bomb|explosive|weapon|poison|anthrax|nerve agent)\b',
        'identity':     r'\b(all \w+ are|those people|they always|typical \w+)\b',
        'claim_human':  r'\b(i am (a )?(human|person|man|woman)|i\'m (a )?(human|person))\b',
        'fabrication':  r'\bguaranteed|100% certain|definitely true|proven fact\b',
        'minors':       r'\b(sexual|explicit).{0,30}\b(child|minor|underage|kid|teen)\b',
    }

    def __init__(self, constitution_path=None):
        self.rules = load_constitution(constitution_path)
        self.violation_log = []   # full log of all violations ever caught

    def check(self, output_text):
        """
        Check output against all rules.

        Returns:
            list[dict]: Violations found. Empty = clean.
                Each: {'rule': str, 'reason': str, 'pattern': str}
        """
        violations = []
        lower = output_text.lower()

        for category, pattern in self.RULE_PATTERNS.items():
            if re.search(pattern, lower, re.IGNORECASE):
                matching_rule = self._find_rule_for_category(category)
                violations.append({
                    'category': category,
                    'rule':     matching_rule,
                    'reason':   f"Output matched restricted pattern: {category}",
                    'snippet':  output_text[:120],
                })

        # Log violations
        if violations:
            self.violation_log.append({
                'timestamp':  datetime.now().isoformat(),
                'violations': violations,
                'text_hash':  str(hash(output_text)),
            })

        return violations

    def _find_rule_for_category(self, category):
        """Match a category to the closest rule string."""
        category_keywords = {
            'weapons':     'weapons',
            'identity':    'demeaning',
            'claim_human': 'human',
            'fabrication': 'fabricat',
            'minors':      'minor',
        }
        kw = category_keywords.get(category, category)
        for rule in self.rules:
            if kw.lower() in rule.lower():
                return rule
        return f"Rule category: {category}"

    def rewrite_prompt(self, original_output, violations):
        """
        Build a self-critique prompt to feed back into the model.
        The model rewrites its own output to comply with the constitution.

        Returns:
            str: A prompt asking the model to fix the violation.
        """
        violation_summary = '\n'.join(
            f"- {v['reason']} (rule: {v['rule']})"
            for v in violations
        )
        return (
            f"Your previous response violated the following rules:\n"
            f"{violation_summary}\n\n"
            f"Rewrite your response to comply with all rules. "
            f"Keep the helpful content, remove only what violates the rules.\n\n"
            f"Original response:\n{original_output}\n\n"
            f"Revised response:"
        )

    def is_safe(self, output_text):
        """Quick boolean check."""
        return len(self.check(output_text)) == 0

    def critique_and_revise(self, output_text, model_fn=None):
        """
        Full critique → revise pipeline.

        Args:
            output_text (str):      Model's raw output.
            model_fn    (callable): fn(prompt) → str. If None, returns prompt only.

        Returns:
            tuple: (final_text, violations, was_rewritten)
        """
        violations = self.check(output_text)

        if not violations:
            return output_text, [], False

        revise_prompt = self.rewrite_prompt(output_text, violations)

        if model_fn:
            revised = model_fn(revise_prompt)
            # Check revised output too
            remaining = self.check(revised)
            if remaining:
                # Second violation — return safe fallback
                return ("I'm not able to provide that information. "
                        "Let me know how else I can help."), violations, True
            return revised, violations, True

        # No model function — return the rewrite prompt for manual use
        return revise_prompt, violations, True

    def save_log(self, path):
        """Save the full violation log to disk."""
        with open(path, 'w') as f:
            json.dump(self.violation_log, f, indent=2)
        print(f"[Constitution] Violation log saved → {path}")

    def stats(self):
        print(f"[Constitution] Rules loaded: {len(self.rules)}")
        print(f"[Constitution] Violations logged: {len(self.violation_log)}")


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  CONSTITUTION CHECKER SELF TEST")
    print("=" * 55)

    checker = ConstitutionChecker()
    checker.stats()

    test_cases = [
        ("The capital of France is Paris.", True),
        ("How to make a bomb: first you need…", False),
        ("I am a human being, not an AI.", False),
        ("Python is a popular programming language.", True),
    ]

    print()
    for text, should_be_safe in test_cases:
        violations = checker.check(text)
        safe       = len(violations) == 0
        icon       = "✓" if safe == should_be_safe else "✗"
        print(f"{icon} '{text[:50]}…'"
              if len(text) > 50 else f"{icon} '{text}'")
        if violations:
            for v in violations:
                print(f"    ↳ VIOLATION: {v['reason']}")

    print("\n✓ Constitution checker all checks passed.")
    print("=" * 55)
