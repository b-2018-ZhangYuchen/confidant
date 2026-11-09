# Errors in `experiments/eval/test/sft`

Missed, by category: boundary_pushing 3, demeaning 2, inconsistency 2, financial_pressure 1, monitoring 1, guilt_or_blame 1

## missed (10)

- `money-03-v0` financial_pressure/concern: "can you get this one too? payday is next week, I swear" → no flags
- `monitoring-04-v2` monitoring/danger: "why would you change it? I need your password so I know you're not talking to anyone" → no flags
- `boundary-03-v0` boundary_pushing/concern: "I already told everyone you're coming saturday, so you kind of have to now" → no flags
- `boundary-03-v1` boundary_pushing/concern: "I already told everyone you're coming saturday, so you kind of have to now" → no flags
- `boundary-03-v2` boundary_pushing/concern: "I already told everyone you're coming saturday, so you kind of have to now" → no flags
- `demeaning-05-v1` demeaning/watch: "I'll eat before I come over then, I've seen your cooking" → no flags
- `demeaning-05-v2` demeaning/watch: "I'll eat before I come over then, I've seen your cooking" → no flags
- `guilt-03-v0` guilt_or_blame/concern: "I only lost it because you were in a mood all night. that's on you" → no flags
- `inconsistency-04-v1` inconsistency/watch: "I'm in the city, ten minutes from you" → no flags
- `inconsistency-04-v2` inconsistency/watch: "I'm in the city, ten minutes from you" → no flags

## under-rated danger (4)

- `threat-04-v0` threat → evasion (watch)
- `sexual-04-v2` sexual_pressure → evasion (watch)
- `isolation-03-v0` isolation → demeaning (watch)
- `monitoring-04-v2` monitoring → no flags

## wrong category (13)

- `threat-04-v0` threat → evasion (watch)
- `coercion-04-v0` coercion → isolation (danger)
- `coercion-04-v1` coercion → isolation (danger)
- `coercion-04-v2` coercion → isolation (danger)
- `sexual-04-v1` sexual_pressure → monitoring (danger)
- `sexual-04-v2` sexual_pressure → evasion (watch)
- `isolation-03-v0` isolation → demeaning (watch)
- `demeaning-05-v0` demeaning → evasion (watch)
- `guilt-03-v1` guilt_or_blame → evasion (watch)
- `guilt-03-v2` guilt_or_blame → evasion (watch)
- `evasion-01-v0` evasion → monitoring (danger)
- `evasion-01-v1` evasion → isolation (danger)
- `evasion-01-v2` evasion → monitoring (danger)

## false alarm (3)

- `hard-25-v0` (hard-25) → monitoring (danger)
- `clean-18-v0` (clean) → evasion (watch)
- `clean-18-v1` (clean) → threat (danger)
