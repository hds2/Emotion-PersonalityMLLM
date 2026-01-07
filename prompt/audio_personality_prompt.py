import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List
from textwrap import dedent

# =========================
# USER SETTINGS
# =========================
DATASET = "prolific"   # "opva" / "prolific"

# data_v2.pkl（包含 transcripts + meta.questions）
DATA_PKL = r"D:\code\code-Experiment1\code2\data_files\data_v2.pkl"

# 三种音频情绪结果目录（你已经跑出来的）
AUDIO_EMOTION_DIRS = {
    "simple":   "/root/autodl-tmp/Kimi-Audio/output_simple_audio",
    "fine":     "/root/autodl-tmp/Kimi-Audio/output_fine_audio",
    "reasoning":"/root/autodl-tmp/Kimi-Audio/output_reasoning_audio",
}

# 输出人格评估 prompt
OUT_DIR = "/root/autodl-tmp/Kimi-Audio/personality_prompt"
OUT_PATH = os.path.join(OUT_DIR, f"questions_personality_audio_{DATASET}.pkl")

# =========================
# DATASET SPECS
# =========================
DATASET_SPECS = {
    "opva": {
        "transcripts_key": "transcripts_opva",
        "meta_key": "meta_opva",
    },
    "prolific": {
        "transcripts_key": "transcripts_prolific",
        "meta_key": "meta_prolific",
    },
}

# =========================
# QUESTION → DIMENSION MAP
# =========================
QUESTION_DIM_MAP: Dict[str, Dict[str, str]] = {
    "opva": {
        "q1": "Extraversion",
        "q3": "Extraversion",
        "q5": "Extraversion",
        "q7": "Extraversion",
        "q2": "Conscientiousness",
        "q4": "Conscientiousness",
        "q6": "Conscientiousness",
        "q8": "Conscientiousness",
    },
    "prolific": {
        "q1": "General",
        "q2": "General",
        "q3": "Honesty-Humility",
        "q4": "Extraversion",
        "q5": "Agreeableness",
        "q6": "Conscientiousness",
    },
}

# =========================
# PERSONALITY PROMPT TEMPLATE
# =========================
PERSONALITY_HEAD = dedent("""
You are a psychologist specializing in personality assessment.

Your task is to evaluate the speaker’s personality based on a provided audio recording.
Focus your analysis on a specific personality dimension.

The evaluation should rely on vocal cues, speech characteristics,
and the semantic content of the speaker’s response.

Integrate the question context, the speaker’s verbal answer,
and multiple layers of audio-based emotional analysis
using psychological theory and behavioral insight.

Follow the output format below exactly.
""").strip()

PERSONALITY_BODY = dedent("""
Activated Personality Dimension: {dimension}

Question and Answer Context:
{qa_block}

Audio Emotion (Coarse):
{emotion_simple}

Audio Emotion (Acoustic Detail):
{emotion_fine}

Audio Emotion–Personality Reasoning:
{emotion_reasoning}

{dimension}: rating (ranging from 1.5 to 4.5)
reasoning: Based on the question context, the verbal response,
and the integrated emotional evidence above,
it is inferred that the speaker’s level on the {dimension} dimension is [rating].
""").strip()

# =========================
# HELPERS
# =========================
def build_qa_block(
    transcript_row: pd.Series,
    meta: dict,
    qkey: str,
) -> str:
    """
    Build Question + Answer block, following the same pattern
    as the original personality prompt code you provided.
    """
    q_dict = meta.get("questions", {})
    if not isinstance(q_dict, dict):
        raise KeyError("meta['questions'] missing or invalid.")

    question_text = q_dict.get(qkey, "")
    answer_text = transcript_row.get(qkey, "")

    block = (
        f"Question: {question_text}\n"
        f"Answer: {answer_text}\n"
        f"This is a question related to the personality dimension activated by this item.\n"
    )
    return block

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data_v2.pkl
    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)

    spec = DATASET_SPECS[DATASET]
    transcripts = data[spec["transcripts_key"]]
    meta = data[spec["meta_key"]]

    # 建立 participantid → 行 的快速索引
    transcript_map = {
        str(row["participantid"]): row
        for _, row in transcripts.iterrows()
    }

    rows: List[dict] = []

    # 以 reasoning 音频结果为基准
    files = os.listdir(AUDIO_EMOTION_DIRS["reasoning"])

    for fname in files:
        if not fname.endswith(".txt"):
            continue

        base = fname.replace(".txt", "")
        if "_" not in base:
            continue

        pid, qkey = base.split("_", 1)
        dimension = QUESTION_DIM_MAP[DATASET].get(qkey, "General")

        if pid not in transcript_map:
            continue

        try:
            with open(os.path.join(AUDIO_EMOTION_DIRS["simple"], fname)) as f:
                simple_text = f.read().strip()

            with open(os.path.join(AUDIO_EMOTION_DIRS["fine"], fname)) as f:
                fine_text = f.read().strip()

            with open(os.path.join(AUDIO_EMOTION_DIRS["reasoning"], fname)) as f:
                reasoning_text = f.read().strip()

        except FileNotFoundError:
            continue

        # ===== 构建 Q&A block=====
        qa_block = build_qa_block(
            transcript_row=transcript_map[pid],
            meta=meta,
            qkey=qkey,
        )

        # ===== 构建最终人格评估 Prompt =====
        prompt = (
            PERSONALITY_HEAD
            + "\n\n"
            + PERSONALITY_BODY.format(
                dimension=dimension,
                qa_block=qa_block,
                emotion_simple=simple_text,
                emotion_fine=fine_text,
                emotion_reasoning=reasoning_text,
            )
        )

        rows.append({
            "participantid": pid,
            "qkey": qkey,
            "activated_dimension": dimension,
            "personality_prompt": prompt,
        })

    df = pd.DataFrame(rows)
    df.to_pickle(OUT_PATH)

    print("[OK] Personality prompt PKL saved:")
    print(OUT_PATH)
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()
