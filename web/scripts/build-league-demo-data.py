#!/usr/bin/env python3
"""Build compact static data for the gameplay data demo."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = "https://f005.backblazeb2.com/file/all-game-data-sample-001/manifest.json"
DEFAULT_SESSION_IDS: list[str] = []
KEY_NAMES = {
    "1": "Esc",
    "2": "1",
    "3": "2",
    "4": "3",
    "5": "4",
    "6": "5",
    "7": "6",
    "8": "7",
    "14": "Tab",
    "15": "Q",
    "16": "W",
    "17": "E",
    "18": "R",
    "19": "T",
    "20": "Y",
    "21": "U",
    "22": "I",
    "23": "O",
    "24": "P",
    "28": "Enter",
    "29": "A",
    "30": "S",
    "31": "D",
    "32": "F",
    "33": "G",
    "34": "H",
    "35": "J",
    "36": "K",
    "40": "`",
    "41": "Shift",
    "43": "Z",
    "44": "X",
    "45": "C",
    "46": "V",
    "47": "B",
    "52": "/",
    "55": "Alt",
    "56": "Space",
    "72": "Up",
    "73": "PageUp",
    "74": "Num -",
    "75": "Left",
    "82": "Insert",
}
BUTTON_NAMES = {"1": "Left", "2": "Right", "3": "Middle"}


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_gzip_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return gzip.decompress(response.read()).decode("utf-8", errors="replace")


def display_game_name(manifest_game: dict[str, str], fallback: str) -> str:
    return manifest_game.get("gameName") or manifest_game.get("game") or fallback


def normalize_event(row: dict[str, str], session_id: str, game: str) -> dict[str, Any] | None:
    event_type = row["event_type"]
    timestamp = int(float(row["t_ms"] or 0))
    input_name = row.get("input_name", "")
    value_a = row.get("value_a", "")
    value_b = row.get("value_b", "")

    if event_type == "cursor_pos":
        if not value_a or not value_b:
            return None
        return {
            "timestampMs": timestamp,
            "type": "mousemove",
            "x": int(float(value_a)),
            "y": int(float(value_b)),
            "sessionId": session_id,
            "game": game,
        }
    if event_type in {"key_down", "key_up"}:
        return {
            "timestampMs": timestamp,
            "type": "keydown" if event_type == "key_down" else "keyup",
            "key": KEY_NAMES.get(input_name, f"code {input_name}"),
            "rawInput": input_name,
            "sessionId": session_id,
            "game": game,
        }
    if event_type in {"mouse_button_down", "mouse_button_up"}:
        return {
            "timestampMs": timestamp,
            "type": "mousedown" if event_type == "mouse_button_down" else "mouseup",
            "button": BUTTON_NAMES.get(input_name, f"button {input_name}"),
            "rawInput": input_name,
            "sessionId": session_id,
            "game": game,
        }
    if event_type == "mouse_wheel":
        return {
            "timestampMs": timestamp,
            "type": "wheel",
            "deltaX": int(float(value_a or 0)),
            "deltaY": int(float(value_b or 0)),
            "sessionId": session_id,
            "game": game,
        }
    return None


def summarize_session(manifest_game: dict[str, str], game_name: str) -> dict[str, Any]:
    data = fetch_json(manifest_game["data"])
    summary = data.get("summary", {})
    details = data.get("details", {})
    recording = summary.get("recording", {}) or {}
    ffprobe = summary.get("ffprobe", {}) or {}
    match = details.get("match", {}) or {}
    participant = match.get("info", {}).get("participants", [{}])[0] if isinstance(match.get("info"), dict) else {}

    events_text = fetch_gzip_text(manifest_game["events"])
    reader = csv.DictReader(io.StringIO(events_text))
    event_counts: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    cursor_bounds = {"minX": None, "maxX": None, "minY": None, "maxY": None}
    last_cursor_bucket = -1

    for row in reader:
        event_type = row["event_type"]
        event_counts[event_type] += 1
        event = normalize_event(row, manifest_game["publicId"], game_name)
        if event is None:
            continue
        if event["type"] == "mousemove":
            timestamp = event["timestampMs"]
            bucket = timestamp // 250
            if bucket == last_cursor_bucket:
                continue
            last_cursor_bucket = bucket
            x = event["x"]
            y = event["y"]
            cursor_bounds["minX"] = x if cursor_bounds["minX"] is None else min(cursor_bounds["minX"], x)
            cursor_bounds["maxX"] = x if cursor_bounds["maxX"] is None else max(cursor_bounds["maxX"], x)
            cursor_bounds["minY"] = y if cursor_bounds["minY"] is None else min(cursor_bounds["minY"], y)
            cursor_bounds["maxY"] = y if cursor_bounds["maxY"] is None else max(cursor_bounds["maxY"], y)
        normalized.append(event)

    normalized.sort(key=lambda event: event["timestampMs"])
    duration_ms = int(recording.get("recordedDurationMs") or (ffprobe.get("durationS") or 0) * 1000)
    event_total = sum(event_counts.values())
    champion = participant.get("championName") or details.get("roles", {}).get("championName")
    queue = summary.get("queueType") or details.get("extraData", {}).get("queueType")

    return {
        "id": manifest_game["publicId"],
        "game": game_name,
        "gameType": manifest_game.get("gameType"),
        "gameName": manifest_game.get("gameName") or game_name,
        "source": manifest_game.get("source"),
        "description": f"{game_name} gameplay capture with synchronized input telemetry.",
        "recordedAt": summary.get("gamePlayedAt"),
        "durationMs": duration_ms,
        "eventCount": event_total,
        "normalizedEventCount": len(normalized),
        "video": manifest_game["video"],
        "dataUrl": manifest_game["data"],
        "eventsUrl": manifest_game["events"],
        "resolution": {
            "width": ffprobe.get("width") or recording.get("outputWidth"),
            "height": ffprobe.get("height") or recording.get("outputHeight"),
        },
        "fps": ffprobe.get("avgFrameRate") or summary.get("videoFramerate"),
        "codec": ffprobe.get("codec"),
        "videoSizeBytes": ffprobe.get("sizeBytes"),
        "captureMethod": recording.get("captureMethod"),
        "droppedFrames": recording.get("droppedFrames"),
        "laggedFrames": recording.get("laggedFrames"),
        "sessionIndex": recording.get("sessionIndex"),
        "champion": champion,
        "queue": queue,
        "rawEventCounts": dict(event_counts),
        "cursorBounds": cursor_bounds,
        "events": normalized,
    }


def summarize_clip(manifest_game: dict[str, str], game_name: str) -> dict[str, Any]:
    try:
        data = fetch_json(manifest_game["data"])
    except Exception:
        data = {}
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    recording = summary.get("recording", {}) or {}
    ffprobe = summary.get("ffprobe", {}) or {}
    return {
        "id": manifest_game["publicId"],
        "video": manifest_game["video"],
        "dataUrl": manifest_game["data"],
        "eventsUrl": manifest_game["events"],
        "game": game_name,
        "gameType": manifest_game.get("gameType"),
        "gameName": manifest_game.get("gameName") or game_name,
        "source": manifest_game.get("source"),
        "recordedAt": summary.get("gamePlayedAt"),
        "durationMs": int(recording.get("recordedDurationMs") or (ffprobe.get("durationS") or 0) * 1000),
        "resolution": {
            "width": ffprobe.get("width") or recording.get("outputWidth"),
            "height": ffprobe.get("height") or recording.get("outputHeight"),
        },
        "fps": ffprobe.get("avgFrameRate") or summary.get("videoFramerate"),
        "codec": ffprobe.get("codec"),
        "videoSizeBytes": ffprobe.get("sizeBytes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default="web/static/data/league-demo.json")
    parser.add_argument("--sessions", nargs="*", default=DEFAULT_SESSION_IDS)
    args = parser.parse_args()

    manifest = fetch_json(args.manifest)
    by_id = {game["publicId"]: game for game in manifest["games"]}
    selected_games = [by_id[session_id] for session_id in args.sessions] if args.sessions else manifest["games"]
    fallback_game_name = manifest.get("game", "Gameplay")
    sessions = [
        summarize_session(game, display_game_name(game, fallback_game_name))
        for game in selected_games
    ]
    with ThreadPoolExecutor(max_workers=16) as executor:
        clip_library = list(
            executor.map(
                lambda game: summarize_clip(game, display_game_name(game, fallback_game_name)),
                manifest["games"],
            )
        )

    output = {
        "sourceManifest": args.manifest,
        "bucket": manifest.get("bucket"),
        "baseUrl": manifest.get("baseUrl"),
        "game": manifest.get("game"),
        "source": manifest.get("source"),
        "totalSessions": manifest.get("count"),
        "filesPerSession": manifest.get("filesPerGame"),
        "generatedFromSessions": [game["publicId"] for game in selected_games],
        "generatedAt": "2026-06-17",
        "clipLibrary": clip_library,
        "notes": [
            "events.csv.gz is normalized into timestamped keyboard, mouse button, wheel, and sampled mousemove events.",
            "High-frequency mouse_delta events are counted but omitted from the interactive page to keep the static payload small.",
            "Cursor overlay uses raw desktop cursor coordinates normalized into the sampled cursor bounding box."
        ],
        "sessions": sessions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
