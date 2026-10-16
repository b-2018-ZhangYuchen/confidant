"""Tests for the chat-app export adapters. No network, no API key needed."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from confidant.cli import main
from confidant.ingest.exports import (
    DateOrder,
    ExportFormat,
    detect_format,
    parse_export,
    read_export,
)
from confidant.ingest.transcript import TranscriptError, read_transcript
from confidant.models import Role

WHATSAPP = "examples/whatsapp_chat.txt"
IMESSAGE = "examples/imessage_chat.txt"

ANDROID = """\
2/3/26, 7:04 PM - Messages and calls are end-to-end encrypted. Tap to learn more.
2/3/26, 7:04 PM - Robin: hey! your profile said you bike to work
2/3/26, 7:31 PM - Sam: the river path, mostly
2/3/26, 7:33 PM - Robin: <Media omitted>
2/14/26, 9:00 AM - Robin: happy valentine's, no pressure
"""


# -- detection -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[14/06/2026, 18:02:11] Ari: hi", ExportFormat.WHATSAPP),
        ("\ufeff\u200e[14/06/2026, 18:02:11] Ari: hi", ExportFormat.WHATSAPP),
        ("2/3/26, 7:04 PM - Robin: hey", ExportFormat.WHATSAPP),
        ("14.06.26, 18:02 - Ari: hallo", ExportFormat.WHATSAPP),
        ("Jul 03, 2026  8:12:45 PM\nMe\nhi", ExportFormat.IMESSAGE),
        ('[{"chats": [{"body": "hi", "timestamp": "2026-01-01 10:00:00"}]}]', ExportFormat.HINGE),
        # Confidant's own bracketed format has no comma between date and time.
        ("[2026-03-02 19:04] Robin: hey", ExportFormat.CONFIDANT),
        ("[03/02/2026 19:04] Robin: hey", ExportFormat.CONFIDANT),
        ("# owner: Sam\nRobin: hey", ExportFormat.CONFIDANT),
    ],
)
def test_formats_are_told_apart(text, expected):
    assert detect_format(text) is expected


# -- WhatsApp --------------------------------------------------------------------


def test_whatsapp_ios_export_reads_into_a_conversation():
    imported = read_export(WHATSAPP, owner="Noor")
    conversation = imported.conversation
    assert imported.format is ExportFormat.WHATSAPP
    assert (conversation.owner_name, conversation.match_name) == ("Noor", "Ari")
    first = conversation.messages[0]
    assert first.role is Role.MATCH
    assert first.text.startswith("hi! it's Ari from the climbing gym")
    assert first.timestamp == datetime(2026, 6, 14, 18, 2, 11)
    assert conversation.source == WHATSAPP


def test_whatsapp_notices_media_and_deletions_are_left_out_and_counted():
    imported = read_export(WHATSAPP, owner="Noor")
    texts = [m.text for m in imported.conversation]
    # iOS files the encryption notice under the other person's name.
    assert not any("end-to-end encrypted" in t for t in texts)
    assert not any("omitted" in t or "deleted" in t for t in texts)
    assert imported.left_out == {"media": 1, "deleted": 1, "system": 1}
    assert imported.note() == (
        "read as a WhatsApp export, 10 messages; "
        "left out 1 media message, 1 deleted message and 1 app notice."
    )


def test_whatsapp_multiline_messages_and_edit_markers():
    texts = [m.text for m in read_export(WHATSAPP, owner="Noor").conversation]
    assert "I am! I want to try the orange one on the overhang\nthat one looked mean" in texts
    assert "ok that is very encouraging actually" in texts


def test_whatsapp_android_twelve_hour_month_first():
    imported = parse_export(ANDROID, owner="Sam")
    stamps = [m.timestamp for m in imported.conversation]
    # 2/14 can only be month-first, which settles 2/3 as February 3rd.
    assert stamps == [
        datetime(2026, 2, 3, 19, 4),
        datetime(2026, 2, 3, 19, 31),
        datetime(2026, 2, 14, 9, 0),
    ]
    assert imported.left_out == {"system": 1, "media": 1}


def test_whatsapp_order_settles_dates_when_only_one_reading_is_chronological():
    text = "[01/02/2026, 10:00:00] Ari: first\n[02/01/2026, 10:00:00] Noor: hey\n"
    stamps = [m.timestamp for m in parse_export(text, owner="Noor").conversation]
    # Only month-first keeps these in order: Jan 2nd then Feb 1st.
    assert stamps == [datetime(2026, 1, 2, 10), datetime(2026, 2, 1, 10)]


def test_whatsapp_ambiguous_dates_are_asked_about_not_guessed():
    text = "[01/02/2026, 10:00:00] Ari: hi\n[01/02/2026, 10:05:00] Noor: hey\n"
    with pytest.raises(TranscriptError, match="--dates dmy"):
        parse_export(text, owner="Noor")
    day_first = parse_export(text, owner="Noor", dates="dmy").conversation
    assert day_first.messages[0].timestamp == datetime(2026, 2, 1, 10)
    month_first = parse_export(text, owner="Noor", dates=DateOrder.MDY).conversation
    assert month_first.messages[0].timestamp == datetime(2026, 1, 2, 10)


def test_whatsapp_dates_that_do_not_fit_the_given_order_are_refused():
    with pytest.raises(TranscriptError, match="do not all read as mdy"):
        parse_export("[14/06/2026, 18:02:11] Ari: hi\n", owner="Noor", dates="mdy")


def test_whatsapp_needs_to_be_told_who_the_owner_is():
    with pytest.raises(TranscriptError, match=r"--owner NAME .*'Ari', 'Noor'"):
        read_export(WHATSAPP)


def test_whatsapp_owner_who_never_speaks_is_an_error():
    with pytest.raises(TranscriptError, match="does not speak"):
        read_export(WHATSAPP, owner="Sam")


def test_whatsapp_group_chats_are_refused():
    text = (
        "[14/06/2026, 18:02:11] Ari: hi all\n"
        "[14/06/2026, 18:03:00] Noor: hey\n"
        "[14/06/2026, 18:04:00] Lee: hello\n"
    )
    with pytest.raises(TranscriptError, match="more than two people"):
        parse_export(text, owner="Noor")


def test_whatsapp_match_overrides_the_name_in_the_export():
    conversation = read_export(WHATSAPP, owner="Noor", match="Ari P").conversation
    assert conversation.match_name == "Ari P"
    assert conversation.match_messages[0].sender == "Ari"


def test_an_export_with_only_media_has_nothing_to_read():
    with pytest.raises(TranscriptError, match="no text messages"):
        parse_export("[14/06/2026, 18:41:03] Ari: \u200eimage omitted\n", owner="Noor")


# -- iMessage ---------------------------------------------------------------------


def test_imessage_export_knows_whose_side_is_whose():
    imported = read_export(IMESSAGE)
    conversation = imported.conversation
    assert imported.format is ExportFormat.IMESSAGE
    assert conversation.owner_name == "Me"
    assert conversation.match_name == "+15555550188"
    assert [m.role for m in conversation] == [
        Role.MATCH,
        Role.OWNER,
        Role.MATCH,
        Role.MATCH,
        Role.OWNER,
        Role.MATCH,
    ]
    assert conversation.messages[0].timestamp == datetime(2026, 7, 3, 20, 12, 45)
    assert conversation.messages[-2].timestamp == datetime(2026, 7, 4, 10, 5, 9)


def test_imessage_owner_and_match_rename_both_sides():
    conversation = read_export(IMESSAGE, owner="Kai", match="Dev").conversation
    assert (conversation.owner_name, conversation.match_name) == ("Kai", "Dev")
    assert conversation.owner_messages[0].sender == "Me"


def test_imessage_tapbacks_attachments_and_receipts_are_not_messages():
    imported = read_export(IMESSAGE)
    texts = [m.text for m in imported.conversation]
    assert "good. you shouldn't" in texts
    assert not any("Loved by" in t or "HEIC" in t or "Read by" in t for t in texts)
    assert imported.left_out == {"media": 1, "tapback": 1}
    assert "ok that's dedication.\nwant to swap notes" in texts[-2]


def test_imessage_announcements_without_a_sender_are_notices():
    text = (
        "Jul 03, 2026  8:12:45 PM\nYou named the conversation books\n\n"
        "Jul 03, 2026  8:13:00 PM\nMe\nhi\n\n"
        "Jul 03, 2026  8:14:00 PM\nDev\nhello\n"
    )
    imported = parse_export(text)
    assert len(imported.conversation) == 2
    assert imported.left_out == {"system": 1}


# -- Hinge ----------------------------------------------------------------------------


def test_hinge_download_is_recognized_and_explained(tmp_path):
    matches = [
        {"match": [{"timestamp": "2026-05-01 10:00:00"}], "chats": []},
        {
            "like": [{"timestamp": "2026-05-02 09:00:00"}],
            "match": [{"timestamp": "2026-05-02 10:00:00"}],
            "chats": [
                {"body": "hi! that hike photo", "timestamp": "2026-05-02 10:05:00"},
                {"body": "which trail was it?", "timestamp": "2026-05-02 10:06:00"},
            ],
        },
    ]
    path = tmp_path / "matches.json"
    path.write_text(json.dumps(matches), encoding="utf-8")
    with pytest.raises(TranscriptError) as excinfo:
        read_export(path)
    message = str(excinfo.value)
    assert "Hinge data download" in message
    assert "only the 2 messages you sent, to 1 match," in message


# -- the rest of Confidant sees the same thing ---------------------------------------


def test_read_transcript_reads_exports_too():
    assert read_transcript(WHATSAPP, owner="Noor").match_name == "Ari"


def test_confidant_transcripts_get_no_note():
    assert read_export("examples/sample_chat.txt").note() is None


def test_zip_exports_get_a_hint(tmp_path):
    path = tmp_path / "WhatsApp Chat - Ari.zip"
    path.write_bytes(b"PK\x03\x04\xff\xfe\x00binary")
    with pytest.raises(TranscriptError, match="unzip it"):
        read_export(path)


def test_cli_stats_on_an_export_says_how_it_was_read(capsys):
    assert main(["stats", WHATSAPP, "--owner", "Noor"]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("Noor <-> Ari\n")
    assert "  messages         10 (4 you / 6 them)" in out
    assert err == (
        f"confidant: {WHATSAPP}: read as a WhatsApp export, 10 messages; "
        "left out 1 media message, 1 deleted message and 1 app notice.\n"
    )


def test_cli_dates_flag_reaches_the_parser(tmp_path, capsys):
    path = tmp_path / "chat.txt"
    path.write_text(
        "[01/02/2026, 10:00:00] Ari: hi\n[01/02/2026, 10:05:00] Noor: hey\n", encoding="utf-8"
    )
    assert main(["stats", str(path), "--owner", "Noor"]) == 2
    assert "--dates dmy" in capsys.readouterr().err
    assert main(["stats", str(path), "--owner", "Noor", "--dates", "dmy"]) == 0
    assert "last message     2026-02-01 10:05" in capsys.readouterr().out


def test_cli_add_saves_an_export_and_saving_it_again_adds_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CONFIDANT_DB", str(tmp_path / "confidant.db"))
    assert main(["add", "Ari", WHATSAPP, "--owner", "Noor"]) == 0
    capsys.readouterr()
    assert main(["add", "Ari", WHATSAPP, "--owner", "Noor"]) == 0
    assert "already saved" in capsys.readouterr().out.lower()
    assert main(["show", "Ari"]) == 0
    assert "  messages         10 (4 you / 6 them)" in capsys.readouterr().out
