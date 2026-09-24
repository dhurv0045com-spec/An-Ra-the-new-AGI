# SHORTCUT BASELINES — audit results

## Status: AUDIT_DESIGNED_NOT_RUN_ON_REAL_DATA

The shortcut baseline framework exists in `citadel_tpu/data_gate.py`
(`gate_shortcut_resistance`) and the 12 baselines are defined, but has NOT
been executed against real production training data because no such data
is materialized.

## Baselines implemented in the gate
- latest_position
- nearest_position
- lexical_overlap
- answer_token_presence
- bag_of_words
- constant_label
- candidate_attestation
- operation_name_lookup
- template_id_lookup
- serialization_position
- prompt_length
- family_classifier

## Known shortcut risks from previous audit
- Latest-position: relevant fact is always last in revision mode
- Lexical overlap: answer appears verbatim in the key fact line
- Constant label: missing_information always queries absent entity
- Candidate attestation: distractors never appear in context
- Template identity: operation name deterministically encodes structure

## Finding
Every currently available synthetic cognition family has at least one
shortcut that a trivial baseline can exploit. This is why T1D showed
TEST exact 0-6.6% (below the 22.5% null) — the model may have learned
shortcuts rather than the intended cognitive skill.
