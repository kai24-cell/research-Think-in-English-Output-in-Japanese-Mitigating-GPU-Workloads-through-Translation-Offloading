#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research translation-output language audit (Japanese expected).

Input: the original paired_results.csv. No API calls and no modifications to input.
Outputs: all records' language flags, unique review queue, suspected non-Japanese,
          reproducible spot-check sample and summary.

Optional higher-confidence detection:
    py -m pip install lingua-language-detector

A language classification is NOT proof of fidelity or answer quality; never silently
remove flagged rows from the research denominators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def is_kana(c: str) -> bool:
    x = ord(c)
    return (0x3040 <= x <= 0x309F) or (0x30A0 <= x <= 0x30FF) or (0x31F0 <= x <= 0x31FF)


def is_han(c: str) -> bool:
    x = ord(c)
    return ((0x3400 <= x <= 0x4DBF) or (0x4E00 <= x <= 0x9FFF)
            or (0xF900 <= x <= 0xFAFF) or (0x20000 <= x <= 0x2FA1F))


def is_hangul(c: str) -> bool:
    x = ord(c)
    return 0xAC00 <= x <= 0xD7AF or 0x1100 <= x <= 0x11FF


def script_stats(s: str) -> dict[str, int]:
    return {
        "kana": sum(is_kana(c) for c in s),
        "han": sum(is_han(c) for c in s),
        "latin": sum(("A" <= c <= "Z") or ("a" <= c <= "z") for c in s),
        "hangul": sum(is_hangul(c) for c in s),
        "nonspace": sum(not c.isspace() for c in s),
    }


def new_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        return None
    # Japanese/Chinese/English/Korean + common western languages.
    return LanguageDetectorBuilder.from_languages(
        Language.JAPANESE, Language.CHINESE, Language.ENGLISH,
        Language.KOREAN, Language.FRENCH, Language.SPANISH, Language.GERMAN,
    ).build()


def detect(s: str, detector) -> tuple[str, dict[str, float]]:
    if detector is None:
        return "UNAVAILABLE", {}
    vals = detector.compute_language_confidence_values(s[:12000])
    scores = {v.language.name: float(v.value) for v in vals}
    if not vals:
        return "UNKNOWN", scores
    return vals[0].language.name, scores


def classify(s: str, detector) -> dict[str, str | float | int]:
    t = (s or "").strip()
    st = script_stats(t)
    language, score = detect(t, detector) if t else ("EMPTY", {})
    ja, zh, en, ko = (score.get(k, 0.0) for k in ("JAPANESE", "CHINESE", "ENGLISH", "KOREAN"))
    reason = ""
    label = "REVIEW_UNCERTAIN"

    if not t:
        reason = "空テキスト"
    elif st["hangul"] >= 4 and st["hangul"] >= st["kana"] * 2:
        label, reason = "SUSPECT_NON_JA", "韓国語文字が優勢"
    elif st["kana"] == 0 and st["han"] >= 12 and detector is not None and zh >= 0.80 and zh - ja >= 0.40:
        label, reason = "SUSPECT_NON_JA", "かな無しで中国語判定が優勢"
    elif st["kana"] == 0 and st["latin"] >= 40 and st["han"] < 5 and detector is not None and en >= 0.8:
        label, reason = "SUSPECT_NON_JA", "日本語文字がほぼ無く英語判定が優勢"
    elif st["kana"] == 0:
        label, reason = "REVIEW_UNCERTAIN", "かな無し（固有名詞や漢字だけの正当な日本語もあり得る）"
    else:
        # Check for a long untranslated segment embedded in Japanese prose.
        suspect_parts = []
        for part in re.split(r"[\r\n。！？!?；;]+", t):
            ps = script_stats(part)
            if ps["nonspace"] < 20 or ps["kana"] >= 2:
                continue
            if detector is None:
                if ps["han"] >= 12:
                    suspect_parts.append("かな無し・漢字中心の長い部分")
                elif ps["latin"] >= 80 and ps["han"] <= 2:
                    suspect_parts.append("英語中心の長い部分")
            else:
                plang, pv = detect(part, detector)
                if ps["han"] >= 12 and pv.get("CHINESE", 0) >= 0.86 and pv.get("JAPANESE", 0) <= 0.12:
                    suspect_parts.append("中国語判定の長い部分")
                elif ps["latin"] >= 80 and ps["han"] <= 2 and plang == "ENGLISH":
                    suspect_parts.append("英語判定の長い部分")
        if suspect_parts:
            label, reason = "REVIEW_MIXED", "/".join(sorted(set(suspect_parts)))
        elif detector is not None and zh > 0.85 and zh - ja > 0.40:
            label, reason = "REVIEW_MIXED", "かなを含むが中国語判定が優勢"
        elif st["kana"] < 3 and st["nonspace"] >= 30:
            label, reason = "REVIEW_UNCERTAIN", "長文に対してかなが極端に少ない"
        elif detector is not None and language not in ("JAPANESE",) and ja < 0.50:
            label, reason = "REVIEW_UNCERTAIN", "文字は日本語を含むが言語判定が割れている"
        else:
            label, reason = "AUTO_JA_LIKELY", "日本語のかなを含み他言語優勢の兆候なし"

    return {
        "language_audit_label": label,
        "language_audit_reason": reason,
        "detector_language": language,
        "detector_ja_score": round(ja, 4) if detector is not None else "",
        "detector_zh_score": round(zh, 4) if detector is not None else "",
        "detector_en_score": round(en, 4) if detector is not None else "",
        "detector_ko_score": round(ko, 4) if detector is not None else "",
        "kana_chars": st["kana"],
        "han_chars": st["han"],
        "latin_chars": st["latin"],
        "hangul_chars": st["hangul"],
        "text_chars": len(t),
        "language_audit_hash": hashlib.sha256(t.encode("utf-8")).hexdigest()[:20],
    }


AUDIT_FIELDS = [
    "model_key", "prompt_id", "repeat_id", "translator_key", "status", "text",
    "language_audit_label", "language_audit_reason", "detector_language",
    "detector_ja_score", "detector_zh_score", "detector_en_score",
    "detector_ko_score", "kana_chars", "han_chars", "latin_chars",
    "hangul_chars", "text_chars", "language_audit_hash",
]


def write_csv(path: Path, rows: list[dict], fields_if_empty=None) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else (fields_if_empty or AUDIT_FIELDS)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="翻訳結果の言語監査（無料・元CSV非変更）")
    parser.add_argument("--input", default="input/paired_results.csv")
    parser.add_argument("--output", default="language_audit_qwen_mistral")
    parser.add_argument("--spotcheck-per-group", type=int, default=5)
    args = parser.parse_args()
    src, out = Path(args.input), Path(args.output)
    with src.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"model_key", "prompt_id", "repeat_id", "translator_key", "text"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"必須CSV列が不足: {required - set(reader.fieldnames or [])}")
        original = list(reader)
    out.mkdir(parents=True, exist_ok=True)
    detector = new_detector()
    print("言語判定モデル:", "Lingua (ローカルCPU / API不要)" if detector else "文字種のみ（Lingua未インストール）")
    if detector is None:
        print("判定の精度を上げるなら: py -m pip install lingua-language-detector")
    cache = {}
    audit_rows = []
    for item in original:
        t = (item.get("text") or "").strip()
        if t not in cache:
            cache[t] = classify(t, detector)
        row = {k: item.get(k, "") for k in ("model_key", "prompt_id", "repeat_id", "translator_key", "status", "text")}
        row.update(cache[t])
        audit_rows.append(row)
    audit_unique = list({row["language_audit_hash"]: row for row in audit_rows}.values())
    reviews = [r for r in audit_unique if r["language_audit_label"] in ("REVIEW_UNCERTAIN", "REVIEW_MIXED")]
    nonja = [r for r in audit_unique if r["language_audit_label"] == "SUSPECT_NON_JA"]
    groups = defaultdict(list)
    for row in audit_unique:
        if row["language_audit_label"] == "AUTO_JA_LIKELY":
            groups[(row["model_key"], row["translator_key"])].append(row)
    rng = random.Random(42)
    spotcheck = []
    for group in sorted(groups):
        eligible = groups[group]
        spotcheck.extend(rng.sample(eligible, min(args.spotcheck_per_group, len(eligible))))
    summary = []
    for (model, trans), rows in sorted(groupby_rows(audit_rows).items()):
        counts = Counter(r["language_audit_label"] for r in rows)
        summary.append({"model_key": model, "translator_key": trans, "rows": len(rows), **counts})
    write_csv(out / "language_audit_all.csv", audit_rows)
    write_csv(out / "language_review_unique.csv", reviews)
    write_csv(out / "language_suspect_non_ja_unique.csv", nonja)
    write_csv(out / "language_spotcheck_unique.csv", spotcheck)
    write_csv(out / "language_summary.csv", summary, ["model_key", "translator_key", "rows"])
    count = Counter(r["language_audit_label"] for r in audit_rows)
    print(f"入力{len(original):,}件 / 判定対象ユニーク文章{len(cache):,}種類")
    print("レコード数別:", dict(count))
    print(f"要確認: {len(reviews)}種類 / 非日本語疑い: {len(nonja)}種類 / 抜き取り: {len(spotcheck)}種類")
    print("保存先:", out.resolve())
    print("※元CSVは変更せず、フラグ付けのみ。自動削除・Sol採点条件の変更は行いません。")


def groupby_rows(rows):
    result = defaultdict(list)
    for r in rows:
        result[(r["model_key"], r["translator_key"])].append(r)
    return result


if __name__ == "__main__":
    main()
