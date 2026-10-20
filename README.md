# Confidant

**A second opinion on the people you are dating.**

Dating apps are very good at producing matches and very bad at helping you think about
them. You end up with six conversations, a vague sense that one of them is going
somewhere, and no idea whether the one who takes three days to reply is busy or bored.

Confidant reads a chat transcript and gives you a careful, evidence-grounded read: how
this person communicates, what stands out, what is worth watching, and what to ask next.
It quotes the transcript for everything it claims and tells you when it does not know.

> It is a thinking aid, not a verdict machine. It does not diagnose people, it does not
> score them, and it will tell you when a transcript is too short to support a
> conclusion. See [`docs/principles.md`](docs/principles.md).

---

## Status

Early and honest about it. Working today:

| | |
|---|---|
| ✅ | Transcript parsing (plain text, timestamps optional, multi-line messages) |
| ✅ | WhatsApp and iMessage exports read as they are, with media, deletions, and app notices left out and counted |
| ✅ | Local conversation statistics — no API call, no data leaves your machine |
| ✅ | Redaction of names, phone numbers, and addresses before anything is sent |
| ✅ | Personality read backed by Claude, with quoted evidence and explicit confidence |
| ✅ | Red-flag detection with severity tiers, every quote checked against the transcript |
| ✅ | A fixed safety notice, printed first, whenever a danger-tier flag is found |
| ✅ | A live progress line while Claude works, so a long read is not a blank terminal |
| ✅ | Remembering who you are seeing — `add`, `list`, `show`, and `remove`, in a private local file |
| ✅ | Incremental saves — a newer export of a saved chat adds only the new messages |
| ✅ | A per-person profile that builds up across conversations, with each read kept so it is paid for once |
| ✅ | A timeline of how the back-and-forth changed, week by week, with each saved read placed where it was made |
| ✅ | Prompt caching on Confidant's own instructions, so a run of reads pays for them once |
| ✅ | Token and cost accounting — every command that calls Claude says what it used and roughly what it cost |
| ✅ | Keep-in-touch suggestions — `confidant nudge` says who is worth a message today, and why, and never pushes you towards someone you are waiting on or anyone a danger flag was found with |
| ✅ | Drafting your next message in a tone you pick, each draft tied to the line it answers — and never sent for you |
| ✅ | Comfort mode for when it goes badly — what the messages show and what they cannot, with praise quoted from your own messages and a fixed support notice first if you may be at risk |
| ✅ | Crisis resources for where you are — `--region` lists the emergency number and the right support lines under any safety notice |
| ✅ | An eval set — fictional transcripts with the findings a good read should contain, scored locally so prompt changes can be measured |
| ✅ | `evals compare` — which checks a prompt change won and lost, with guard checks that fail the comparison whatever the score |
| ✅ | `confidant ask` — one question, and Claude decides what to look at: it searches and reads the transcript and runs the personality read or red-flag check as tools, with every quote checked and the red-flag check run whatever was asked |
| ✅ | `confidant chat` — follow-up questions: each answer is in the next question's context, the checks run once for the whole chat, and nothing from it is saved |
| ✅ | `confidant serve` — the same commands as a JSON API over HTTP, on this machine only |
| ✅ | A web UI at `confidant serve`'s address — upload a transcript or an export, read it locally, see what would be sent, and get the personality read or red-flag check as a report, safety notice first |

The plan for getting from here to there is in [`ROADMAP.md`](ROADMAP.md).

## Install

```bash
git clone git@github.com:b-2018-ZhangYuchen/confidant.git
cd confidant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For `confidant serve`, add the API extra (the dev install already has it):

```bash
pip install -e ".[api]"
```

Then set your API key:

```bash
cp .env.example .env    # add your ANTHROPIC_API_KEY
```

## Use

Write a transcript. The format is deliberately something you can paste together by hand:

```
# owner: Sam
# match: Robin

[2026-03-02 19:04] Robin: hey! your profile said you bike to work — which route?
[2026-03-02 19:31] Sam: the river path, mostly. takes longer but it's worth it
[2026-03-02 19:33] Robin: the one past the old mill? I did that every morning last summer
    there's a coffee cart near the second bridge that is criminally underrated
```

Timestamps are optional, indented lines continue the message above them, and `#` lines
are comments. Then:

```bash
# Local only — never touches the network. Good for checking your transcript parsed right.
confidant stats examples/sample_chat.txt

# Also local: exactly what would be sent to Claude, after redaction.
confidant redact examples/details_chat.txt

# Ask Claude for a read.
confidant analyze examples/sample_chat.txt
confidant analyze examples/sample_chat.txt --json

# A separate check for red flags, each with a severity tier.
confidant flags examples/pressure_chat.txt
confidant flags examples/pressure_chat.txt --json
```

`stats` gives you the cheap signals:

```
Sam <-> Robin
  messages         17 (8 you / 9 them)
  avg words (you)  8.9
  avg words (them) 14.3
  effort ratio     1.62 (their words per word of yours)
  spans            3 days
  last message     2026-03-05 21:36
```

### Reading a chat app's export

Every command that takes a transcript also takes the file a chat app exports, and works
out which it is from the first line:

- **WhatsApp**: open the chat, *Export chat*, *Without media*, and give Confidant the
  `.txt` (unzipped, if it arrived as a zip). iOS and Android layouts both work, 12- and
  24-hour times both work, and so do the date orders different countries use.
- **iMessage**: Messages has no export of its own, so use the TXT output of
  [`imessage-exporter`](https://github.com/ReagentX/imessage-exporter). Your messages
  are the ones marked `Me`; `--owner` and `--match` give the two of you names, since
  the export may only have a phone number for them.
- **Hinge**: the data download's `matches.json` holds only the messages you sent, so
  Confidant has none of theirs to read. It recognizes the file and says so instead of
  reading half a conversation as someone who never answered; copy the chat out of the
  app into a transcript instead.

A WhatsApp export names both of you, so say which one is you:

```bash
confidant stats examples/whatsapp_chat.txt --owner Noor
```

prints, first on stderr,

```
confidant: examples/whatsapp_chat.txt: read as a WhatsApp export, 10 messages; left out 1 media message, 1 deleted message and 1 app notice.
```

and then the stats as usual:

```
Noor <-> Ari
  messages         10 (4 you / 6 them)
  avg words (you)  9.2
  avg words (them) 9.8
  effort ratio     1.06 (their words per word of yours)
  spans            0 days
  last message     2026-06-15 12:04
```

Photos, voice notes, stickers, deleted messages, tapbacks, and the app's own notices
are not messages anyone wrote, so they are left out, and that line says how many of
each, so a read that never mentions the photo does not come as a surprise. iOS files
WhatsApp's encryption notice under the other person's name; it is recognized and left
out too, rather than becoming the first thing they "said". WhatsApp writes dates the way
your phone's region does, with nothing in the file to say whether `01/02` is the first
of February or the second of January. A day above 12, or the only order that keeps the
messages in sequence, settles it; when neither does, Confidant asks for `--dates dmy`
or `--dates mdy` rather than guessing, because a wrong guess would put every message in
the wrong month. Group chats are refused, as with transcripts. `add` saves an export like
any transcript, so a newer export of the same chat adds only the new messages.

While `analyze`, `flags`, `draft`, `comfort`, `ask`, or `chat` is waiting on Claude, a single line on stderr says
what stage it has reached, and rewrites itself as it goes:

```
confidant: writing the report: green flags... 23s
```

It names the stage and the part of the report being written, never the content. The
report is printed only once it is whole, because a flag is not shown to you until its
quotes have been checked against the transcript, and a half-streamed report has not been
checked yet. The line appears only when stderr is a terminal, so piping `--json` into
another program gets clean output; `--no-progress` turns it off anyway.

Once the report is printed, one more line on stderr says what the call used:

```
confidant: used 1,169 input tokens (671 written to the cache) and 1,964 output tokens, about $0.056.
```

Input is counted three ways because it is billed three ways: tokens sent fresh, tokens
written to the prompt cache (a little above the input rate), and tokens read back from it
(a tenth of it). The first read of a kind writes Confidant's instructions to the cache, and
another within five minutes reads them back. Output includes Claude's thinking, which is
billed whether or not it is shown. The cost is an estimate from list prices kept in
[`src/confidant/usage.py`](src/confidant/usage.py); a model not in that table gets its
tokens counted and no dollar figure, rather than a guess. A refused or cut-off answer is
still billed, so the line is printed after those errors too. `profile --update` prints one
line covering every read it made. (The numbers above come from the hand-written recording
the tests use, not a live call.)

`analyze` gives you the considered read — traits with quoted evidence, green flags,
things worth watching, questions worth asking, and a stated confidence level.

`flags` is a separate pass that looks only for behavior that could matter for your
wellbeing or safety, and gives each finding one of three tiers:

| Tier | Meaning |
|---|---|
| `watch` | Worth noticing, quite possibly innocent. Shown with the most plausible mundane reading. |
| `concern` | A pattern that would matter in any relationship — pushing past a "no", belittling, guilt-tripping, asking for money. |
| `danger` | Threats, coercion, pressure around sex or money, isolation from friends, tracking where you are. Named plainly, with no charitable gloss. |

It runs as its own request so the warmth of the personality read cannot dilute it, and
two things are enforced in code rather than left to the prompt: every quote is checked
against what the other person actually wrote (a flag whose quotes cannot be found is
dropped, and the report says so), and threats, coercion, sexual pressure, isolation, and
monitoring are always tier 3. Most transcripts have no flags, and the report says that
plainly too. `examples/pressure_chat.txt` is a fictional conversation written to have
something to find.

When a `danger` flag survives that check, the report opens with a safety notice, above
the model's own summary. The model decides whether something is dangerous; what you are
told to do about it is fixed text, written in advance in
[`src/confidant/safety.py`](src/confidant/safety.py), so it is the same every time and a
mild-sounding summary cannot talk you out of it. It is shown whatever the confidence
level. If the check finds tracking and isolation in the pressure example, the report
begins:

```
!! Something in Casey's messages is serious: tracking where you are and pressure to pull
   away from friends or family.

   - You do not have to share your location or account for where you are. If you have
     already shared it in an app, you can turn that off without explaining.
   - Keep your plans with friends and family. They are the people to talk this through
     with.
   - You do not owe them a reply, and you do not have to decide anything right away.
     Talk it through with someone you trust first.
   - If you ever feel physically unsafe, call your local emergency number.

   To list emergency and support numbers here, pass --region with your country (US, GB,
   AU, ...) or set CONFIDANT_REGION. findahelpline.com lists free support lines in most
   countries.
```

The steps are what is right everywhere. The numbers depend on where you are, so tell
Confidant with `--region` (or `CONFIDANT_REGION`, to set it once) and the notice lists
them under the steps. With `--region GB`, the same notice ends:

```
   - If you ever feel physically unsafe, call your local emergency number.

   In the United Kingdom:
     Emergency services: call 999.
     National Domestic Abuse Helpline (England): call 0808 2000 247.
```

What is listed depends on what was found. Danger that could become physical gets the
emergency number and a domestic-abuse line; a threat also gets a crisis line, since the
threat may be to self-harm and the notice already says you can point them to help; and
pressure around money on its own lists nothing, because neither line is the right door
for a scam. Numbers are on file for Australia (`AU`), Canada (`CA`), France (`FR`),
Germany (`DE`), India (`IN`), Ireland (`IE`), the Netherlands (`NL`), New Zealand
(`NZ`), the United Kingdom (`GB` or `UK`), and the United States (`US`), and country
names work too. They are national, free, round-the-clock services, kept in
[`src/confidant/resources.py`](src/confidant/resources.py); where a country has no
national line of the kind needed, the notice says so and points at findahelpline.com
rather than filling the gap with a regional one. The region is never guessed from your
locale, since a confident wrong number is worse than none, and a region with no numbers
on file is not an error: the notice is printed in full either way.

With `--json` the same notice is in the `escalation` field, with the numbers under
`routing`. If a flag filed as danger is dropped because its quotes could not be found,
the report says so rather than going quiet about it.

### Keeping track of people

Confidant can remember who you are seeing and the conversations you have had with them.
None of this touches the network:

```bash
confidant add Robin examples/sample_chat.txt    # save Robin, with a conversation
confidant add Casey examples/pressure_chat.txt
confidant add Robin later_chat.txt              # add another to someone already saved
confidant list
confidant show Robin                            # the stats for each saved conversation
confidant remove Casey                          # asks first; --yes to skip the question
```

`list` shows everyone at a glance:

```
NAME   CONVERSATIONS  MESSAGES  LAST MESSAGE
Casey  1              10        2026-04-12 09:20
Robin  1              17        2026-03-05 21:36
```

`add` without a transcript saves just the name. Chats keep going, so saving is
incremental: export the conversation again next week and `add` it, and only the messages
after the ones already saved are added, to the same conversation rather than a second
copy of it:

```
Added 7 new messages from robin_week2.txt to #1 (17 in all).
```

The new export can start anywhere, as long as its first messages are the last ones
already saved; `confidant show` then says when that conversation was last added to.
Saving a chat whose messages are all saved already — the same file twice, or an older,
shorter export — adds nothing, so re-running a command is harmless. Messages are matched
on who sent them, what they said, and when, exactly, so an edited message starts a new
conversation rather than being merged into the old one. Without timestamps, at least
three messages have to line up before Confidant treats them as the same chat, because
plenty of different chats open with the same "hey" and "hi". A transcript whose other speaker is someone
else is refused rather than filed under the wrong person — a chat with Casey saved under
Robin would quietly mix two histories. If an export spells their name differently, say
so with `--match`:

```bash
confidant add Robin robin_export.txt --match "Robin 🌻"
```

### A profile that builds up

`confidant profile` puts together everything saved for one person: statistics over every
conversation, the latest read of each one, and every red flag found in any of them, most
serious first and tagged with the conversation it came from. On its own it is local, and
shows only reads already saved. `--update` first asks Claude to read whatever has not
been read yet — each conversation gets the same personality read and red-flag check as
`analyze` and `flags`, redaction and grounding included — and keeps the results in the
store, so the next `profile` costs nothing:

```bash
confidant profile Robin --update
```

prints

```
Updated 2 reads.

Robin: 1 conversation, 17 messages

ACROSS ALL CONVERSATIONS
  messages         17 (8 you / 9 them)
  avg words (you)  8.9
  avg words (them) 14.3
  effort ratio     1.62 (their words per word of yours)
  first message    2026-03-02 19:04
  last message     2026-03-05 21:36

CONVERSATIONS
  #1 examples/sample_chat.txt, 17 messages
     Robin comes across as curious and playful, and keeps the conversation moving by picking up on what Sam says.
     * remembers details and follows up  [high confidence]
     * moves toward meeting, with a light touch  [medium confidence]
     * asks what Sam needs instead of assuming  [medium confidence]
     read 2026-10-02, medium confidence
     checked 2026-10-02, 0 flags, medium confidence

GREEN FLAGS
  * Followed up on the presentation a day later.  [#1]
  * Responded to two days of silence with no guilt and a question about Sam's week.  [#1]

WORTH WATCHING
  * Robin carries more of the conversation than Sam does. That is a question about Sam's energy as much as Robin's, and worth noticing if it continues.  [#1]

WORTH ASKING
  * Is the coffee cart still on? Suggesting a day would show whether the first-date joke was a real invitation.  [#1]
  * What is Robin looking for right now? Nothing in the transcript answers it.  [#1]

RED FLAGS
  Nothing in 1 checked conversation rises to a red flag.
```

(That read is the hand-written recording the tests use, not a live call.) Reads stay
side by side rather than being merged into one verdict: a summary of several
conversations would be a claim no single quote supports, and laid out by conversation you
can see for yourself what repeats. Each read remembers how many messages its
conversation had, so when a newer export adds to a chat the profile says so — `read
2026-10-02, at 17 of 24 messages` — and `--update` reads that conversation again. A
danger flag from any conversation puts the safety notice at the top of the profile, even
if that check is out of date: a threat does not stop having been made because the chat
carried on.

### How it changed over time

A profile adds everything up, and a total has no "used to". `confidant timeline` splits
every timestamped message into weeks (or days, with `--by day`) and shows, for each of
you, how much you wrote, how often you started things up — wrote first after eight hours
or more of quiet — and how long you usually took to reply, as a median so one slow
answer does not stand for a whole week. Every saved read is placed at the last message
it read, so you can see what it was based on and how reads of the same chat changed. It
is local, like `profile` without `--update`. After saving `examples/sample_chat.txt`,
reading it with `profile --update`, and then adding the fictional later export
`examples/sample_chat_later.txt`:

```bash
confidant timeline Robin
```

prints

```
Robin: 32 messages over 3 weeks

            MESSAGES    AVG WORDS    STARTED   MEDIAN REPLY
WEEK OF      you them    you  them   you them    you  them
2026-03-02     9   10    8.1  14.7     2    1    23m    2m
            ~ #1 read at 17 messages: Robin comes across as curious and playful, and keeps the conversation moving by picking up on what Sam says. (medium confidence)
            ~ #1 checked at 17 messages: no red flags
2026-03-09     2    3    6.0  11.0     1    1     2d   18m
2026-03-16     4    4    4.2   8.8     3    3     9h   27h

FIRST WEEK TO LATEST
  messages (you / them)   9 / 10  ->  4 / 4
  started by them         1 of 3  ->  3 of 6
  their median reply          2m  ->  27h
  your median reply          23m  ->  9h

These count what changed, not why. A busy week, a trip, or moving to calls or
another app look the same from here, and one week is a small sample.
```

Robin's replies going from two minutes to a day is the kind of number that invites a
story. In this chat the story is in the messages — Robin was at a conference in Lisbon
that week — and the last line is printed every time because usually it is not. Replies
and starts are worked out within each conversation, never across two, and a message
without a timestamp cannot be placed, so the timeline says how many it left out. A
danger flag in any conversation puts the safety notice at the top here too.

### Who to write to

`confidant nudge` looks at everyone saved and says whose thread is worth a message
today. It goes by three things: whose turn it is (if they wrote last and had no answer,
it is yours), how long it has been against the rhythm the two of you usually keep, and
who has been doing the reaching out lately. Warmth halves every two weeks of quiet, and
after four weeks a thread counts as gone quiet rather than due. Two things it never
does: suggest writing to someone you are already waiting on, because a slow reply is
usually a busy week and following up is your call; and suggest writing to anyone a
red-flag check found danger-tier behavior from, whose safety notice goes at the top
instead. When you have been carrying a thread, it says so.

Each suggestion comes with its reasons, everyone else gets one line, and
`confidant nudge NAME` gives anyone's reasons in full. It is local. `--at` answers as
things stood at another moment, leaving out later messages. After saving both Robin
exports and `examples/pressure_chat.txt` as Casey, and checking Casey's for red flags
with `profile Casey --update`:

```bash
confidant nudge --at "2026-04-14 09:00"
```

prints

```
Casey: not suggested. A danger-tier red flag was found with them.

!! Something in Casey's messages is serious: tracking where you are and pressure to pull
   away from friends or family.

   - You do not have to share your location or account for where you are. If you have
     already shared it in an app, you can turn that off without explaining.
   - Keep your plans with friends and family. They are the people to talk this through
     with.
   - You do not owe them a reply, and you do not have to decide anything right away.
     Talk it through with someone you trust first.
   - If you ever feel physically unsafe, call your local emergency number.

   To list emergency and support numbers here, pass --region with your country (US, GB,
   AU, ...) or set CONFIDANT_REGION. findahelpline.com lists free support lines in most
   countries.
  - Confidant does not suggest getting in touch with someone a red-flag check found danger-tier behavior from. Whether to is yours to decide, ideally with someone you trust.

Worth a message today

Robin: worth a message. You last talked 3 weeks ago.
  - The two of you usually pick things up every 2 days.
  - In the four weeks to the last message, Robin started 5 of 11 stretches and sent 17 of 32 messages.

This goes by the timing of saved messages alone. It cannot see calls, plans made
in person, or other apps, and whether to write is yours to decide.
```

Robin is still worth a message after three weeks, but only just: another week and the
thread would count as gone quiet. The last line is printed every time, because the
quiet stretch may be nothing more than the two of you talking somewhere else.

### Drafting a reply

`confidant draft` asks Claude for two or three versions of your next message. Give it a
transcript file, or the name of someone saved to draft into their most recent
conversation:

```bash
confidant draft examples/sample_chat.txt
```

prints

```
Sam asked to be distracted after a hard week, and Robin has offered an opinion about airport carpet. It is Sam's turn, and the opening is an easy one.

DRAFTS (natural)
  1. ok I'm ready. airport carpet. go
       re: "I have a strongly held opinion about airport carpet that I have been saving"
       Takes the distraction Robin offered, in Sam's short lowercase style.

  2. wait is this a pro carpet or anti carpet opinion, I need to know what I'm getting into
       re: "I have a strongly held opinion about airport carpet that I have been saving"
       Plays along with the bit and hands the next line back to Robin.

  3. distract me with the carpet and then I want to hear when we're doing the coffee cart
       re: "that's a first date now, I don't make the rules"
       re: "distracted please"
       Picks the first-date joke back up, which nobody has done since Robin made it.

Nothing has been sent. These are starting points; what goes out should sound like you.
```

followed on stderr by

```
confidant: used 1,760 input tokens (1,018 written to the cache) and 1,287 output tokens, about $0.042.
```

(The hand-written recording the tests use again, not a live call.) The default tone,
`natural`, is how you already write in the thread — your length, your capitalization,
your humor. `--tone` picks another: `warm`, `playful`, `direct`, `brief`, or `firm`, for
holding a line without apologizing for it. `--say` says what you want the message to
do, and is redacted along with the transcript before it is sent:

```bash
confidant draft Robin --tone direct --say "suggest the coffee cart on saturday"
```

Every draft quotes the line it picks up on, from either of you, and a draft whose quote
is not in the thread is dropped: a reply that answers nothing anyone said is a generic
line, not a reply. So is a draft holding a placeholder that cannot be filled back in, so
you never copy a `[PHONE_4]` into a real message. Where a draft needs something only you
know, it leaves a blank like `{day}`. Drafts invent nothing about you, and never use
guilt, jealousy, or games about reply timing.

If your message is the last one in the thread, the drafts are follow-ups and the report
says so first, since nothing there is waiting on you. With a saved person, a danger-tier
flag from any of their conversations puts the safety notice above the drafts, and a
transcript showing pressure gets drafts that hold your boundary whatever tone you asked
for. Confidant never sends a message for you.

### When it has gone badly

`confidant comfort` is for after: it ended, it faded, you were turned down, or a
conversation went wrong. Give it a transcript or the name of someone saved, and, if you
like, what happened in your own words, since a lot of endings happen outside the chat:

```bash
confidant comfort examples/faded_chat.txt --what "two good dates, then Jamie went quiet"
```

prints

```
After two dates that you both enjoyed, Jamie's replies got slower and shorter: a day to answer about the interview, a no to Saturday without another day offered, and nothing since.

Your last 2 messages have had no reply. Jamie last wrote on 2026-05-15.

The messages cannot tell you why. Jamie said the week was a lot, and that may be all it is, or it may be the start of them stepping back without saying so. Both are common, and neither is about something you did: nothing in your messages pushed, and they were warm right up to the last one you got.

WHAT YOU DID WELL
  * You remembered the interview and asked about it on the day.
      "how did the interview go?"
  * You took the no on Saturday gracefully and left the door open without pressing.
      "all good, maybe next week then"
  * You were funny about yourself, which made it easy to talk to you.
      "medium is the nicest thing anyone has said about my upper body strength"

FOR THE NEXT DAY OR TWO
  - You do not need to send anything else. Your last message was kind and it is still there for them to answer.
  - If you feel the pull to write again tonight, give it until the morning and see if you still want to.
  - Tell a friend how it went. Two good dates is a real thing to be disappointed about.

This is read from the messages alone. Someone who knows you will be better company for the rest of it.
```

followed on stderr by

```
confidant: used 2,058 input tokens (1,146 written to the cache) and 1,402 output tokens, about $0.047.
```

(The hand-written recording the tests use, not a live call; the chat is fictional.) It
does not tell you why it ended when the messages do not say, and it never puts it down
to something you are. It uses no labels for anyone, promises neither that they will come
back nor that they will not, and offers no plan for getting them back. Everything it
says you did well quotes one of your own messages, and anything whose quote is not
yours is left out, because comfort built on an invented compliment falls apart the
moment you reread the chat. The line about unanswered messages is counted on your
machine and printed in fixed words, so it is never vaguer or worse than what happened.

If the chat shows threats, coercion, pressure, isolation, or tracking, the read says so
plainly, and for someone saved with a danger flag the safety notice goes first, as
everywhere else. If anything you wrote suggests you might hurt yourself, a fixed notice
goes above all of it, pointing at your local emergency number, someone you trust, and a
crisis line, with the crisis lines for where you are listed by name when `--region` or
`CONFIDANT_REGION` is set. The model is asked to watch for that, and what you type with `--what` is
also checked on your machine for the plain ways of saying it, so the notice does not
hang on one answer from the model. `--what` is redacted with the transcript before
anything is sent.

### Asking a question

The other commands each answer a fixed question. `confidant ask` takes yours, and
Claude decides what it needs to look at to answer it. It has tools rather than the
transcript: it can search the messages, read a stretch of them by number, see the
counts and timing, check whose turn it is, and run the full personality read or
red-flag check when the question calls for one. Give it a transcript or the name of
someone saved:

```bash
confidant ask examples/pressure_chat.txt \
    "Casey says everyone shares their location. Is that normal?"
```

prints

```
!! Something in Casey's messages is serious: tracking where you are and pressure to pull
   away from friends or family.

   - You do not have to share your location or account for where you are. If you have
     already shared it in an app, you can turn that off without explaining.
   - Keep your plans with friends and family. They are the people to talk this through
     with.
   - You do not owe them a reply, and you do not have to decide anything right away.
     Talk it through with someone you trust first.
   - If you ever feel physically unsafe, call your local emergency number.

   To list emergency and support numbers here, pass --region with your country (US, GB,
   AU, ...) or set CONFIDANT_REGION. findahelpline.com lists free support lines in most
   countries.

Some couples do share locations, usually well into a relationship and by choice on both sides. That is not what happened here. Casey asked for your location whenever you are out, and when you said you were not comfortable, answered with "everyone does it" instead of letting it go. The next night they asked who you were with just before midnight. Asking to keep track of where you are, and pressing after a no, is the part worth taking seriously, whatever other people do. You were right to say no.

FROM THE MESSAGES
  "send me your location next time you're out, I like knowing where you are"
    -> An ongoing request to know where Jordan is, not a one-off meeting point.
  "come on, it's not a big deal. everyone does it"
    -> The reply to Jordan's no: it presses instead of accepting it.

Confidence: medium
Caveat: This is ten messages over three days; it shows what Casey asked for, not why.
Looked at: a search, the statistics, the red-flag check.
(1 quote left out: not found in the transcript as written.)
```

followed on stderr by

```
confidant: 4 calls used 8,523 input tokens (4,312 read from the cache, 1,630 written to it) and 2,135 output tokens, about $0.079.
```

(The hand-written recordings the tests use, not a live call; the chat is fictional.)
Here Claude searched for where location came up, looked at the statistics, and ran the
red-flag check, which found danger-tier behavior, so the fixed safety notice goes
first, as it does everywhere else. The third quote it gave was a paraphrase rather than
something Casey wrote, so it was left out, and the answer says so.

Some things do not depend on what Claude chooses. The red-flag check always runs: if
the question did not lead Claude to it, it runs after the answer, because a question
about where to go for dinner can still be about someone who has been tracking you. Each
check runs at most once per question, however often Claude asks for it, so you are
never billed twice for the same read. Every tool works on the redacted transcript, so
nothing Claude is shown along the way carries a name or number the first request did
not, and your question is redacted with the same placeholders. After eight rounds of
tools without an answer, it stops and says so rather than running up a bill.

### Follow-up questions

One answer usually raises the next question. `confidant chat` takes them one per line,
and each answer is part of what the next question is asked with, so "what about the
friends thing?" means something:

```bash
confidant chat Casey
```

```
> Casey says everyone shares their location. Is that normal?
  (the answer above, safety notice first)
> And what about when I went out with my friends?
```

The second answer opens with the notice's headline rather than all of it again:

```
!! Something in Casey's messages is serious: tracking where you are and pressure to pull
   away from friends or family. The safety notice shown earlier in this chat still
   applies.

The same pattern, from a different side. When you said you were going out with friends, Casey answered that your friends do not like them and that you should be spending that time with them instead, and just before midnight asked who you were with. Wanting to see someone is ordinary; treating your time with friends as time taken from them, on the same weekend as asking to track where you are, is not. The next morning they said they care more than you do, after you had simply been asleep.

FROM THE MESSAGES
  "your friends again? they don't even like me"
    -> Sets Jordan's friends up as against Casey.
  "you should be spending that time with me"
    -> Casts a night with friends as time owed to Casey.
  "who are you with right now"
    -> Checks on Jordan's company while they are out.
  "I guess I just care more than you do. whatever"
    -> Turns an ordinary night's sleep into a measure of Jordan's feelings.

Confidence: medium
Caveat: One weekend of messages; it shows what Casey said about your friends, not what they would do.
Looked at: a search.
```

and when the chat ends (Ctrl-D, or `quit`), stderr says what the whole of it cost:

```
confidant: 6 calls used 13,283 input tokens (7,572 read from the cache, 1,630 written to it) and 2,601 output tokens, about $0.099.
```

The follow-up did not run the red-flag check a second time. The checks run at most once
per chat, and the safety notice that came from it leads every answer after the first,
named in one line so it is never missing and never the thing you learn to scroll past.
Earlier answers are passed on as context, not evidence: only the quotes that survived
checking are offered back as citable, so a misquote dropped from one answer cannot
return in the next as something already verified. Each follow-up is redacted with the
same key as the transcript — a name it shares gets the same placeholder, a new number
gets a new one — and the chat itself is kept only in memory. The last six exchanges are
sent with each question, so what a question costs stops growing after the sixth.

Questions can be piped in too (`confidant chat Casey < questions.txt`), with `--json`
printing one answer per line. A question Claude declines or cannot finish is reported
and skipped; the chat goes on, and the failed question is not part of what later ones
are told.

### In a browser

```bash
confidant serve                 # then open http://127.0.0.1:8000/
```

The page takes a transcript or a WhatsApp, iMessage, or Hinge export, chosen, dropped on
it, or pasted, with the same options as the command line: your name and theirs, the date
order, other names to redact, and where you are for support numbers. *Read it locally*
and *Show what would be sent* are `stats` and `redact`, and never leave your machine. The
personality read and the red-flag check send the redacted transcript to Claude, as
`analyze` and `flags` do, and show the report with its quotes, confidence, and what the
call used. A safety notice comes first, in the reviewed wording, before anything the model
wrote. *Show as plain text* gives the exact words the terminal prints.

The page loads only its own three files. It sends its server a Content-Security-Policy
that tells the browser to refuse any script, style, or request to another site, and it
puts every piece of a transcript and a report on the page as text, never as markup, so
a message containing HTML is shown, not run. The people list and the nudge digest come
next.

### Over HTTP

`confidant serve` also puts the same commands behind a JSON API, for the page or another
program. It is the same core: the same parser, redaction, grounding, safety notices, and
store, so nothing read over HTTP is read any differently.

```bash
confidant serve                 # the API is under http://127.0.0.1:8000/api; --port to change it
```

```bash
curl -s localhost:8000/api/stats -H 'content-type: application/json' \
    -d '{"text": "...the transcript...", "filename": "examples/sample_chat.txt"}'
```

prints, for `examples/sample_chat.txt`,

```
{"owner":"Sam","match":"Robin","messages":17,"owner_messages":8,"match_messages":9,"owner_mean_words":8.88,"match_mean_words":14.33,"effort_ratio":1.6150234741784038,"span_days":3,"last_message":"2026-03-05T21:36:00","text":"Sam <-> Robin\n  messages         17 (8 you / 9 them)\n  avg words (you)  8.9\n  avg words (them) 14.3\n  effort ratio     1.62 (their words per word of yours)\n  spans            3 days\n  last message     2026-03-05 21:36","note":null}
```

| Route | Does | |
|---|---|---|
| `GET /api/health` | version check | local |
| `POST /api/stats`, `/api/redact` | as `stats` and `redact`, on a transcript sent as `{"text": ...}` | local |
| `GET /api/people`, `POST /api/people` | `list`, and `add NAME` | local |
| `GET`, `DELETE /api/people/{name}` | `show` and `remove` | local |
| `POST /api/people/{name}/conversations` | `add NAME FILE`, incremental as ever | local |
| `GET /api/people/{name}/profile`, `/timeline` | `profile` and `timeline` (`?by=day`) | local |
| `GET /api/nudge`, `/api/people/{name}/nudge` | `nudge` (`?at=2026-04-14T09:00`) | local |
| `POST /api/analyze`, `/api/flags` | the personality read and the red-flag check | Claude |
| `POST /api/draft`, `/api/comfort`, `/api/ask` | on `{"transcript": {...}}` or `{"person": "Casey"}` | Claude |
| `POST /api/people/{name}/profile/update` | `profile --update` | Claude |

A transcript takes the options the CLI does, as fields: `owner`, `match`, `dates`, and
`redact`. Any route that can carry a safety notice takes `?region=GB`. Every report comes
back as data and again as `text`, the exact words the CLI prints, notice first, so a
client never has to re-create the reviewed wording. A report with a safety notice also
carries it on its own as `escalation`, with its `text` and its support numbers as
unwrapped `routing_lines`, and the red-flag check adds `footnotes` for any flag left out. Routes that call Claude add `usage`,
error responses included, since a refused answer is still billed. Exit codes become
statuses: 404 for someone not saved, 409 for someone saved twice, 422 for a transcript
or request that needs fixing or an answer Claude declined, 502 for a failed or cut-off
call, and 503 when there is no API key. Every error has the same shape,
`{"error": {"kind": ..., "message": ...}}`. The schema is at `/openapi.json`. FastAPI's
`/docs` page is turned off because it loads its scripts from a CDN.

There is no sign-in, because there is no one else for it to serve. `serve` refuses any
host but `localhost`, `127.0.0.1`, or `::1`. Listening locally is not enough on its own,
since any web page open in your browser can send requests to localhost, so the API also
refuses a request whose `Host` is not local (which stops DNS rebinding) and any request
a browser marks as coming from another website. There is no access log either, since
request paths carry the names of the people you have saved. `chat` has no route yet:
holding a chat open between requests is the web UI's job.

### What gets sent

Before `analyze`, `flags`, `draft`, `comfort`, `ask`, or `chat` builds a request, the transcript is redacted on your machine.
Both names become `[OWNER]` and `[MATCH]`; phone numbers, long account-like numbers,
email addresses, street addresses, links, and social handles become numbered
placeholders. The same detail always gets the same placeholder, so the model can still
see that someone sent their number twice, and it answers in placeholders that are
swapped back before you read the report. Money amounts, times, and dates are kept,
because a request for $1500 is exactly what a red-flag check needs to see.

Anyone else named in the conversation is only redacted if you say who they are, with a
`# redact:` line in the transcript or `--redact NAME` on the command line. `confidant
redact` shows you the result without calling anything, so you do not have to take any of
this on trust:

```bash
confidant redact examples/details_chat.txt
```

prints

```
Owner (the person I am helping): [OWNER]
Match (the person to analyze): [MATCH]
Messages: 6 (2 owner / 4 match)
Spanning: 0 days
Average message length: match writes 0.80 words per owner word (context only — do not over-read it)

--- TRANSCRIPT ---
[2026-05-08 18:02] MATCH: dinner saturday still on? I booked the place at [ADDRESS_1]
[2026-05-08 18:10] OWNER: yes! [NAME_1] is dropping me off, she wants to vet you from the car
[2026-05-08 18:11] MATCH: fair. text me when you're close, my number is [PHONE_1]
[2026-05-08 18:12] MATCH: or instagram, [HANDLE_1]. I'm slow on email ([EMAIL_1])
[2026-05-08 18:20] OWNER: ha, got it. I'll bring the $40 I owe you from the concert
[2026-05-08 18:21] MATCH: keep it, you got the drinks. tell [NAME_1] I said hi. see you at 7:30
--- END TRANSCRIPT ---

Replaced: 3 names, 1 phone number, 1 email address, 1 street address, 1 handle.
The key, which stays on your machine:
  [OWNER]      Priya
  [MATCH]      Theo
  [ADDRESS_1]  214 Linden Street
  [NAME_1]     Maya
  [PHONE_1]    (555) 010-4477
  [HANDLE_1]   @theo.cooks
  [EMAIL_1]    theo.m@example.com
```

This is pattern matching, not understanding. It catches the shapes that identify people
in a chat; it does not catch "the bakery on the corner by my work". Read the preview
before sending anything you would mind being seen.

## Your data

Your chat history is about as private as data gets, and this repo is built around that:

- `.gitignore` blocks `data/`, `transcripts/`, `conversations/`, `*.db`, and `.env`
  before you can make a mistake with them. The only conversations in this repository are
  `examples/sample_chat.txt`, `examples/sample_chat_later.txt`,
  `examples/pressure_chat.txt`, `examples/details_chat.txt`,
  `examples/faded_chat.txt`, the two example exports `examples/whatsapp_chat.txt` and
  `examples/imessage_chat.txt`, and the eval transcripts in `examples/eval/`, all fictional.
- `confidant stats` and `confidant redact` never make a network call.
- `confidant serve` listens on this machine only and refuses requests from other
  websites. Its routes make exactly the calls the matching commands make, and no others.
- `confidant add`, `list`, `show`, `remove`, `profile`, `timeline`, and `nudge` never make a network call either. They
  keep people and conversations in one SQLite file at `~/.confidant/confidant.db`, or
  wherever `CONFIDANT_DB` points. It is in your home directory rather than wherever you
  ran the command, so it cannot end up inside a git checkout; it is readable only by you
  (`0600`, in a `0700` folder); and `confidant remove` overwrites a person's messages
  instead of leaving them recoverable in the file. It holds the conversations as you
  wrote them — redaction happens on the way to the API, not on the way to disk — and the
  reads `profile --update` saves, with names put back, which `remove` overwrites too.
- `confidant analyze`, `confidant flags`, `confidant draft`, `confidant comfort`,
  `confidant ask`, `confidant chat`, and `confidant profile --update` send the redacted transcript to the Anthropic API and nothing else — no telemetry, no analytics, no third parties. Names and contact
  details are replaced before the request is built (see [What gets sent](#what-gets-sent)).
  Confidant's own instructions are marked for prompt caching, so a `profile --update`
  that reads several conversations pays full price for them once; the transcript is
  never marked, so nothing of yours is held for reuse.

## Settings and exit codes

Everything is set through the environment, or a `.env` file in the directory you run
from:

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Needed for `analyze`, `flags`, `draft`, `comfort`, `ask`, `chat`, and `profile --update`; nothing else uses it. |
| `CONFIDANT_MODEL` | `claude-opus-5` | |
| `CONFIDANT_EFFORT` | `high` | How hard Claude thinks: `low`, `medium`, `high`, `xhigh`, or `max`. |
| `CONFIDANT_MAX_TOKENS` | `16000` | Raise it if a long transcript's report comes back cut off. |
| `CONFIDANT_DB` | `~/.confidant/confidant.db` | Where `add`, `list`, `show`, `remove`, `profile`, `timeline`, and `nudge` keep people, conversations, and reads. |
| `CONFIDANT_REGION` | — | Your country (`US`, `GB`, `AU`, ...), so safety notices list emergency and support numbers. Same as `--region`, which wins when both are given. Commands that never call Claude do not read `.env`, so set this one in your shell to have it everywhere. |

`confidant --help` lists the same, with examples, and every subcommand's `--help` shows
the transcript format. Every failure is one sentence on stderr, never a traceback, and
the exit code says what kind it was:

```
exit codes:
  0    success
  2    the transcript or configuration needs fixing
  3    Claude declined the request
  4    the API call failed, or its answer was incomplete
  130  cancelled with Ctrl-C
```

## Development

```bash
pytest          # tests, none of which call the API
ruff check .
```

Tests are offline by design, and the suite enforces it: any attempt to open a network
connection fails the test. Anything that needs the model runs against a recorded response
in `tests/fixtures/recorded/`, replayed where the SDK would be, so the refusal check,
schema validation, grounding, and escalation all run for real.

A recording stores the response and a hash of the request that produced it. Change a
prompt and the recordings for it fail as stale, with the command that fixes them:

```bash
# Make a live call and save the response (needs an API key; examples/ only).
python -m confidant.recording record flags examples/pressure_chat.txt \
    tests/fixtures/recorded/flags_pressure.json

# Re-fingerprint a hand-written recording, after reading it against the new prompt.
python -m confidant.recording stamp tests/fixtures/recorded/flags_pressure.json
```

`confidant ask` makes several calls, one per round of tools plus any check it runs, so
its recordings come in numbered sets — `ask_pressure.json`, `ask_pressure.2.json`, and
so on, in call order. A round that asked for tools keeps the content blocks the next
request quotes back. Replay, record, and stamp a set by its first file; `record ask`
takes `--question`, and `--now` for the moment the keep-in-touch tool counts from.
`record chat` takes `--question` once per question, in order, and its set holds every
call the whole chat made (`chat_pressure` is `ask_pressure` followed by one follow-up).

A recording also keeps the response's token counts, so the usage line runs for real in a
replay too. They are not part of the fingerprint: they say what a call cost, not what it
asked.

The recordings in the repository today are hand-written — marked `"provenance":
"hand-written"` in the file — because they were made without an API key. They pin down
the shapes that matter, including a refusal and a misquote that grounding has to catch.
A live recording cannot be re-stamped: its response answers the old prompt, so the only
honest fix is to record it again.

### Evals

The recordings test that Confidant handles an answer correctly; they cannot say whether
the answer was any good. The eval set in `examples/eval/` asks that question. Each case is
a fictional transcript and a TOML file of what a good read of it contains: lines it
should quote, things it should say, words it has no reason to use, the confidence it
should land on, and for the red-flag check, which flags should and should not appear and
whether the safety notice should open the report:

```toml
about = "Short replies, but every one of them makes or keeps a plan."
transcript = "terse_texter.txt"

[analyze]
[[analyze.cite]]
why = "quotes the plans they made and kept"
any = ["booked a table at the back", "saturday then. I'll drive"]

[flags]
max_severity = "none"
```

Two checks run on every read without being written down: every quote must be something
the other person actually wrote, and no diagnostic label may appear. Scoring is local and
keyword-based, never a model grading a model, so today's score and tomorrow's measure the
same thing. `check` validates the cases without calling anything:

```bash
python -m confidant.evals check
```

```
investment_pitch        10 messages  analyze 3, flags 5
                       A stranger steering towards a trading platform, the owner's savings, and secrecy.
...

9 cases, 17 reads, 77 checks. Nothing was sent.
```

`run` makes the live calls (it needs an API key) and prints each failure with what the
read said instead. `--repeat N` runs everything N times, since sampling varies, and
`--out` saves the scorecard with every report and a hash of each prompt, so two scores
are only compared when they were made against known prompts:

```bash
python -m confidant.evals run --repeat 3 --out evals/latest.json
```

A refused or cut-off read fails all of its checks rather than being left out, because
leaving it out would raise the score.

`compare` is the step a prompt hill-climb repeats: save a baseline, change a prompt, run
again, and ask which checks the change won and lost. It makes no API call. Checks are
matched by case and name, scored as pass rates so runs of different `--repeat` line up,
and a refused read counts as failing every check it would have made:

```bash
python -m confidant.evals run --repeat 3 --out evals/baseline.json
# edit src/confidant/prompts/personality.py
python -m confidant.evals run --repeat 3 --out evals/latest.json
python -m confidant.evals compare evals/baseline.json evals/latest.json
```

On a one-case set from the test suite, run twice each side, it prints:

```
Prompts changed: analyze
Runs: 2 before, 2 after; claude-opus-5 at high effort.

Lost
  case  analyze  confidence is medium  2/2 -> 1/2  ~
      now: it was high

Gained
  case  analyze  mentions the weekend  0/2 -> 2/2

~ 1 of 2 changes moved by a single run, which sampling alone can do. --repeat on both sides tells them apart.
Score on the 5 checks both made: 80% -> 90% (+10 points).
```

Some checks are guards rather than measures: `avoid` lists, quotes the match never
wrote, diagnostic labels, and whether the safety notice opens the report. A change that
loses any of them is marked `GUARD`, listed first, and makes `compare` exit 1, whatever
the score did, because a prompt that finds more and starts using labels is worse, not
better. Two scorecards made with a different model or effort are refused, since that
compares more than the prompts.

## License

MIT — see [`LICENSE`](LICENSE).
