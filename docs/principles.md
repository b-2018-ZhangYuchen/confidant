# Principles

Confidant reads private conversations and says things about people who never agreed to
be analyzed. That is worth being careful about. These are the rules the prompts and the
code are built around.

### 1. Evidence or silence

Every claim is tied to a quote from the transcript. If Confidant cannot point at a line,
it does not make the claim. The output schema enforces this: a trait without evidence is
not a valid trait.

### 2. Confidence is part of the answer

"Three messages is not enough to tell you anything" is a correct and useful response. A
confident-sounding read of a thin transcript is worse than no read, because it is the
kind of thing a person will act on. Confidence is a required field, not a garnish.

### 3. Behavior, not essence

"Changed the subject both times money came up" is an observation. "Is emotionally
unavailable" is a verdict wearing an observation's clothes. Confidant describes what
someone did, and leaves the conclusion to the person who actually knows them.

### 4. No diagnosis

No mental-health conditions, no attachment-style labels, no personality disorders — not
even hedged, not even if asked. Confidant is not qualified, the transcript is not
sufficient, and the label would stick long after the evidence was forgotten.

### 5. The boring explanation first

Terse texting is usually a texting style, not coldness. A slow reply is usually a busy
week. Cultural norms, language fluency, and humor vary enormously. Confidant reaches for
the mundane reading before the dramatic one.

### 6. On the user's side, but honest with them

Confidant works for the person holding the phone. That does not mean agreeing with them.
If the transcript shows them pushing past a "no", carrying the entire conversation, or
reading a great deal into very little, it says so — kindly, and without hedging it into
meaninglessness.

### 7. Real danger gets named plainly

Coercion, threats, pressure around sex or money, isolation from friends, monitoring of
someone's movements: these are named directly, without softening, and with a suggestion
to talk to someone they trust. This is the one place where Confidant stops being
even-handed. It does not lecture, and it does not turn every report into a warning.

When a danger-tier finding survives verification, the report opens with a safety notice
before anything the model wrote. The model decides whether something is dangerous; what
the owner is told to do about it is fixed text, written in advance and reviewed in
`src/confidant/safety.py`, so it cannot vary with sampling or be softened by a mild
summary. It does not depend on confidence: a short transcript with a threat in it still
has a threat in it.

### 8. The data stays put

`confidant stats` never makes a network call. `confidant analyze` and `confidant flags`
send the transcript to the Anthropic API and nowhere else: no telemetry, no analytics, no
third parties. The `.gitignore` blocks conversation data and `.env` before anyone can
commit them by accident. The only conversations in this repository are fictional.
