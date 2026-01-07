import pickle
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict
from textwrap import dedent

# =========================
# USER SETTINGS
# =========================
PKL_PATH = r"D:\code\code-Experiment1\code2\data_files\data_v2.pkl"

SELECT_DATASET = "prolific"   # "opva" / "prolific"
QUESTION_TYPE = "factors"     # only used for filename

OUT_BASE = Path(r"D:\code\code-Experiment1\code2")

# =========================
# DATASET SPECS
# =========================
DATASET_SPECS = {
    "opva": {
        "transcripts_key": "transcripts_opva",
        "meta_key": "meta_opva",
        "out_subdir": "opva-video-personality-prompt",
    },
    "prolific": {
        "transcripts_key": "transcripts_prolific",
        "meta_key": "meta_prolific",
        "out_subdir": "prolific-video-personality-prompt",
    },
}

# =========================
# QUESTION → PERSONALITY DIMENSION MAP
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
# VIDEO PERSONALITY PROMPT TEMPLATE
# =========================
PERSONALITY_HEAD = dedent("""
You are a psychologist specializing in personality assessment.

Your task is to evaluate the individual’s personality based on a video recording.
Focus your analysis on a specific personality dimension.

The evaluation should integrate:
- The semantic content of the individual’s verbal response
- The emotional and expressive cues visible in the video
- The temporal evolution of facial expressions and nonverbal behavior

Use psychological theory and behavioral insight to support your judgment.

Please follow the output format below exactly.
""").strip()

PERSONALITY_BODY = dedent("""
Activated Personality Dimension: {dimension}

Question and Answer Context:
{qa_block}

Video-based Emotional Evidence:
{video_emotion}

{dimension}: rating (ranging from 1.5 to 4.5)
reasoning: Based on the question context, the verbal response,
and the observed emotional and nonverbal behavior in the video,
it is inferred that the individual’s level on the {dimension} dimension is [rating].
""").strip()

# =========================
# HELPERS
# =========================
def get_question_keys(meta: dict) -> List[str]:
    q_dict = meta.get("questions", {})
    if not isinstance(q_dict, dict) or len(q_dict) == 0:
        raise KeyError("meta['questions'] missing or empty")
    return sorted(q_dict.keys())


def build_qa_block(
    transcript_row: pd.Series,
    meta: dict,
    qkey: str,
) -> str:
    """
    Build Question + Answer block (learned from your original personality code).
    """
    q_dict = meta.get("questions", {})
    question_text = q_dict.get(qkey, "")
    answer_text = transcript_row.get(qkey, "")

    return (
        f"Question: {question_text}\n"
        f"Answer: {answer_text}\n"
        f"This question is designed to probe the {qkey} context.\n"
    )

# =========================
# BUILD DATAFRAME
# =========================
def build_video_personality_df(
    transcripts: pd.DataFrame,
    meta: dict,
    dataset: str,
) -> pd.DataFrame:
    qkeys = get_question_keys(meta)
    dim_map = QUESTION_DIM_MAP[dataset]

    rows = []
    for _, row in transcripts.iterrows():
        pid = str(row["participantid"])

        for qk in qkeys:
            dimension = dim_map.get(qk, "General")

            qa_block = build_qa_block(
                transcript_row=row,
                meta=meta,
                qkey=qk,
            )

            
            video_emotion_prompt = (
                "The following is a summary of the emotional and nonverbal "
                "behavior observed in the video when the individual responded "
                "to this question.\n"
                "emotion: [predicted emotional state]\n"
                "reasoning: [based on facial expressions, gaze, posture, and temporal dynamics]"
            )

            prompt = (
                PERSONALITY_HEAD
                + "\n\n"
                + PERSONALITY_BODY.format(
                    dimension=dimension,
                    qa_block=qa_block,
                    video_emotion=video_emotion_prompt,
                )
            )

            rows.append({
                "participantid": pid,
                "qkey": qk,
                "activated_dimension": dimension,
                "personality_prompt": prompt,
            })

    return pd.DataFrame(rows)

# =========================
# SAVE
# =========================
def save_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)

# =========================
# MAIN
# =========================
def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    if SELECT_DATASET not in DATASET_SPECS:
        raise ValueError(f"Invalid dataset: {SELECT_DATASET}")

    spec = DATASET_SPECS[SELECT_DATASET]
    transcripts = data[spec["transcripts_key"]].copy()
    meta = data[spec["meta_key"]]

    out_dir = OUT_BASE / spec["out_subdir"]

    df = build_video_personality_df(
        transcripts=transcripts,
        meta=meta,
        dataset=SELECT_DATASET,
    )

    out_path = out_dir / f"questions_video_personality_{QUESTION_TYPE}.pkl"
    save_df(df, out_path)

    print(
        f"[OK] Video personality prompts generated | "
        f"Dataset={SELECT_DATASET} | "
        f"Rows={len(df)} | "
        f"Saved={out_path}"
    )

if __name__ == "__main__":
    main()
