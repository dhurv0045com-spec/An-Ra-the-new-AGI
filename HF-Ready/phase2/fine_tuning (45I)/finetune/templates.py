# ============================================================
# FILE: finetune/templates.py
# Prompt template system.
#
# Handles: system injection, few-shot examples, dynamic context,
# and a template library for QA, summarisation, code, planning,
# and conversation.
#
# Every template produces a single formatted string ready to
# feed into the model tokenizer.
# ============================================================

import json, os, re
from datetime import datetime


# ── Base template class ───────────────────────────────────

class PromptTemplate:
    """
    A single reusable prompt format.

    Stores:
      - name:         identifier
      - system:       the system prompt string (may contain {variables})
      - user_prefix:  label before the user turn  (e.g. "### User:")
      - asst_prefix:  label before the AI turn    (e.g. "### Assistant:")
      - shot_sep:     separator between few-shot examples
    """

    def __init__(self, name, system='', user_prefix='### User:',
                 asst_prefix='### Assistant:', shot_sep='\n---\n'):
        self.name        = name
        self.system      = system
        self.user_prefix = user_prefix
        self.asst_prefix = asst_prefix
        self.shot_sep    = shot_sep

    def format(self, user_input, assistant_response='',
               context='', few_shots=None, variables=None):
        """
        Build the full prompt string.

        Args:
            user_input          (str):  The current user message.
            assistant_response  (str):  If provided, append as a training target.
            context             (str):  Optional injected context (RAG, memory…).
            few_shots    (list[dict]):  [{"user":…, "assistant":…}, …]
            variables     (dict):       Fills {placeholders} in system prompt.

        Returns:
            str: The fully formatted prompt.
        """
        parts = []

        # 1. System prompt (with variable substitution)
        system = self.system
        if variables:
            for k, v in variables.items():
                system = system.replace('{' + k + '}', str(v))
        if system:
            parts.append(system.strip())
            parts.append('')

        # 2. Few-shot examples
        if few_shots:
            for shot in few_shots:
                parts.append(f"{self.user_prefix}\n{shot['user'].strip()}")
                parts.append(f"{self.asst_prefix}\n{shot['assistant'].strip()}")
                parts.append(self.shot_sep)

        # 3. Injected context
        if context:
            parts.append(f"[Context]\n{context.strip()}\n")

        # 4. Current user turn
        parts.append(f"{self.user_prefix}\n{user_input.strip()}")

        # 5. Assistant turn (empty during inference, filled during training)
        parts.append(f"{self.asst_prefix}")
        if assistant_response:
            parts.append(assistant_response.strip())

        return '\n'.join(parts)

    def training_pair(self, example):
        """
        Format an example dict for training.
        Returns (prompt, completion) where prompt ends just before
        the answer and completion is the answer itself.
        """
        prompt     = self.format(
            user_input          = example['user'],
            assistant_response  = '',
            context             = example.get('context', ''),
        )
        completion = example['assistant']
        return prompt, completion

    def to_dict(self):
        return {
            'name':        self.name,
            'system':      self.system,
            'user_prefix': self.user_prefix,
            'asst_prefix': self.asst_prefix,
            'shot_sep':    self.shot_sep,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ── Template library ──────────────────────────────────────

class TemplateLibrary:
    """
    Registry of named templates.
    Ships with built-in templates; custom ones can be added/saved.
    """

    BUILTIN = {

        'chat': PromptTemplate(
            name        = 'chat',
            system      = ('You are {name}, a helpful and intelligent AI assistant. '
                           'Today is {date}. Answer clearly and concisely.'),
            user_prefix = '### User:',
            asst_prefix = '### Assistant:',
        ),

        'qa': PromptTemplate(
            name        = 'qa',
            system      = ('You are a precise question-answering system. '
                           'Answer factually. If you do not know, say so.'),
            user_prefix = '### Question:',
            asst_prefix = '### Answer:',
        ),

        'summarize': PromptTemplate(
            name        = 'summarize',
            system      = ('You are an expert summarizer. '
                           'Produce a concise, accurate summary. '
                           'Preserve key facts. Remove fluff.'),
            user_prefix = '### Text to summarize:',
            asst_prefix = '### Summary:',
        ),

        'code': PromptTemplate(
            name        = 'code',
            system      = ('You are an expert software engineer. '
                           'Write clean, correct, well-commented {language} code. '
                           'Return only the code block unless explanation is requested.'),
            user_prefix = '### Task:',
            asst_prefix = '### Code:',
        ),

        'plan': PromptTemplate(
            name        = 'plan',
            system      = ('You are a strategic reasoning system. '
                           'Think step by step. '
                           'Break complex goals into clear, ordered actions.'),
            user_prefix = '### Goal:',
            asst_prefix = '### Plan:',
        ),

        'instruct': PromptTemplate(
            name        = 'instruct',
            system      = 'Follow the instruction below precisely.',
            user_prefix = '### Instruction:',
            asst_prefix = '### Response:',
        ),
    }

    def __init__(self):
        self._templates = dict(self.BUILTIN)   # copy so mutating is safe

    def get(self, name, variables=None):
        """
        Retrieve a template by name.

        Args:
            name      (str):  Template identifier.
            variables (dict): Auto-fill {placeholders}. date is filled automatically.

        Returns:
            PromptTemplate
        """
        tmpl = self._templates.get(name)
        if tmpl is None:
            raise KeyError(f"Unknown template '{name}'. "
                           f"Available: {list(self._templates)}")

        # Always inject {date} automatically
        defaults = {'date': datetime.now().strftime('%Y-%m-%d')}
        if variables:
            defaults.update(variables)

        # Return a copy with system prompt already substituted
        d = tmpl.to_dict()
        system = tmpl.system
        for k, v in defaults.items():
            system = system.replace('{' + k + '}', str(v))
        d['system'] = system
        copy = PromptTemplate(**d)
        return copy

    def register(self, template):
        """Add a custom template to the library."""
        self._templates[template.name] = template

    def save(self, directory):
        """Persist all custom templates (not builtins) to disk."""
        os.makedirs(directory, exist_ok=True)
        custom = {k: v for k, v in self._templates.items()
                  if k not in self.BUILTIN}
        path = os.path.join(directory, 'custom_templates.json')
        with open(path, 'w') as f:
            json.dump({k: v.to_dict() for k, v in custom.items()}, f, indent=2)
        print(f"[Templates] Saved {len(custom)} custom templates → {path}")

    def load(self, directory):
        """Load custom templates from disk."""
        path = os.path.join(directory, 'custom_templates.json')
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for name, d in data.items():
            self._templates[name] = PromptTemplate.from_dict(d)
        print(f"[Templates] Loaded {len(data)} custom templates")

    def list(self):
        return list(self._templates.keys())

    def format(self, name, user_input, **kwargs):
        """Shortcut: get template and format in one call."""
        tmpl = self.get(name, kwargs.pop('variables', None))
        return tmpl.format(user_input, **kwargs)


# Singleton — import and use directly
library = TemplateLibrary()


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  TEMPLATE SYSTEM SELF TEST")
    print("=" * 60)

    lib = TemplateLibrary()

    print("\n── chat template ──")
    chat_tmpl = lib.get('chat', variables={'name': 'AXIOM'})
    prompt = chat_tmpl.format(
        user_input          = 'Explain black holes in 2 sentences.',
        assistant_response  = '',
        few_shots           = [
            {'user':      'What is 2+2?',
             'assistant': '4.'},
        ],
    )
    print(prompt)

    print("\n── code template ──")
    code_tmpl = lib.get('code', variables={'language': 'Python'})
    print(code_tmpl.format('Write a function that reverses a string.'))

    print("\n── training pair ──")
    tmpl  = lib.get('instruct')
    p, c  = tmpl.training_pair({
        'user':      'List 3 planets.',
        'assistant': 'Mars, Venus, Jupiter.',
    })
    print(f"Prompt   : {repr(p[:80])}…")
    print(f"Completion: {repr(c)}")

    print(f"\nAvailable templates: {lib.list()}")
    print("\n✓ Template system all checks passed.")
    print("=" * 60)
