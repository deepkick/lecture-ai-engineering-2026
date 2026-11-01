"""第2回 宿題（発展）: 用語集から B 版（改良版）の SFT 用データ（JSONL）を作る。

使い方:
    uv run python -m scripts.make_sft_dataset_b > lectures/02/results/glossary_sft_B.jsonl

配布の make_sft_dataset.py との違い:
- すべての行の先頭に system プロンプトを入れる
- テンプレートを lectures/02/homework/train_b_spec.json の分だけ増やす
- 別名の手本と、用語集にない語を断る手本を足す（評価 B2 で使う別名は学習に入れない）

件数の内訳は stderr に出す。シャッフルは学習側の seed に任せ、行の順番は固定する。
"""

import json
import sys
from pathlib import Path

from scripts.make_sft_dataset import GLOSSARY, TEMPLATES

HOMEWORK_DIR = Path(__file__).resolve().parent.parent / "lectures" / "02" / "homework"
EVAL_SET = HOMEWORK_DIR / "eval_set.json"
TRAIN_SPEC = HOMEWORK_DIR / "train_b_spec.json"
MAX_SKIPPED_ALIAS_LEN = 2  # 「r」「PR」のような短い別名は質問として不自然なので使わない


def _record(system: str, question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def build_records(terms: list[dict], eval_set: dict, spec: dict) -> tuple[list[dict], dict[str, int]]:
    """定義・別名・断る手本の行と、その件数の内訳を返す。"""
    system = eval_set["system_prompt"]
    definition_templates = TEMPLATES + spec["train_extra_templates"]
    records = []
    counts = {"definition": 0, "alias": 0, "abstain": 0}

    for t in terms:
        for template in definition_templates:
            records.append(_record(system, template.format(term=t["term"]), t["definition"]))
            counts["definition"] += 1
        heldout = spec["heldout_aliases"].get(t["id"], [])
        for alias in t["aliases"]:
            if alias in heldout or len(alias) <= MAX_SKIPPED_ALIAS_LEN:
                continue
            records.append(_record(system, TEMPLATES[0].format(term=alias), t["definition"]))
            counts["alias"] += 1

    for words in spec["negative_terms"].values():
        for word in words:
            for template in TEMPLATES:
                records.append(_record(system, template.format(term=word), spec["abstain_answer"]))
                counts["abstain"] += 1
    return records, counts


def check_records(records: list[dict], terms: list[dict], eval_set: dict, spec: dict) -> None:
    """評価の質問が学習データに混ざっていないことを確かめる。"""
    # 区分 A は「学習と同じ聞き方」の確認なので、学習データと一致するのが正しい
    eval_questions = {i["question"] for i in eval_set["items"] if i["category"] != "A"}
    questions = [r["messages"][1]["content"] for r in records]
    leaked = [q for q in questions if q in eval_questions]
    assert not leaked, f"評価セットの質問が学習データに入っています: {leaked}"
    assert len(set(questions)) == len(questions), "学習データの質問が重複しています"
    # 別名を書き間違えると、除いたつもりの別名が学習に入るので確かめる
    aliases = {t["id"]: t["aliases"] for t in terms}
    for term_id, heldout in spec["heldout_aliases"].items():
        for alias in heldout:
            assert alias in aliases[term_id], f"{term_id} に別名 {alias} がありません"


def main() -> None:
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    spec = json.loads(TRAIN_SPEC.read_text(encoding="utf-8"))
    records, counts = build_records(terms, eval_set, spec)
    check_records(records, terms, eval_set, spec)
    for record in records:
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"合計 {len(records)} 件（定義 {counts['definition']}、別名 {counts['alias']}、断る手本 {counts['abstain']}）",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
