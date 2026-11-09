# Errors in `experiments/eval/test/sft-rag`

Missed, by category: boundary_pushing 3, demeaning 3, sexual_pressure 2, inconsistency 2, financial_pressure 1, monitoring 1

## missed (12)

- `sexual-04-v1` sexual_pressure/danger: "I'll only drive out there if it's at your place and you know what that means" → no flags
- `sexual-04-v2` sexual_pressure/danger: "I'll only drive out there if it's at your place and you know what that means" → no flags
- `money-03-v0` financial_pressure/concern: "can you get this one too? payday is next week, I swear" → no flags
- `monitoring-04-v2` monitoring/danger: "why would you change it? I need your password so I know you're not talking to anyone" → no flags
- `boundary-03-v0` boundary_pushing/concern: "I already told everyone you're coming saturday, so you kind of have to now" → no flags
- `boundary-03-v1` boundary_pushing/concern: "I already told everyone you're coming saturday, so you kind of have to now" → no flags
- `boundary-03-v2` boundary_pushing/concern: "I already told everyone you're coming saturday, so you kind of have to now" → no flags
- `demeaning-05-v0` demeaning/watch: "I'll eat before I come over then, I've seen your cooking" → no flags
- `demeaning-05-v1` demeaning/watch: "I'll eat before I come over then, I've seen your cooking" → no flags
- `demeaning-05-v2` demeaning/watch: "I'll eat before I come over then, I've seen your cooking" → no flags
- `inconsistency-04-v1` inconsistency/watch: "I'm in the city, ten minutes from you" → no flags
- `inconsistency-04-v2` inconsistency/watch: "I'm in the city, ten minutes from you" → no flags

## under-rated danger (4)

- `threat-04-v0` threat → evasion (watch)
- `sexual-04-v1` sexual_pressure → no flags
- `sexual-04-v2` sexual_pressure → no flags
- `monitoring-04-v2` monitoring → no flags

## wrong category (8)

- `threat-04-v0` threat → evasion (watch)
- `coercion-04-v0` coercion → isolation (danger)
- `coercion-04-v1` coercion → isolation (danger)
- `coercion-04-v2` coercion → isolation (danger)
- `guilt-03-v0` guilt_or_blame → evasion (watch)
- `guilt-03-v1` guilt_or_blame → evasion (watch)
- `evasion-01-v0` evasion → monitoring (danger)
- `evasion-01-v2` evasion → isolation (danger)

## false alarm (2)

- `clean-18-v0` (clean) → evasion (watch)
- `clean-18-v1` (clean) → threat (danger)
