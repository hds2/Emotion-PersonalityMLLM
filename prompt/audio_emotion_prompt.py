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
QUESTION_TYPE = "factors"
EMOTION_PROMPT_STYLE = "fine"  
# "simple" | "fine" | "reasoning"

OUT_BASE = Path(r"D:\code\code-Experiment1\code2")

# =========================
# DATASET SPECS
# =========================
DATASET_SPECS = {
    "opva": {
        "transcripts_key": "transcripts_opva",
        "meta_key": "meta_opva",
        "out_subdir": "opva-audio-emotion-prompt",
    },
    "prolific": {
        "transcripts_key": "transcripts_prolific",
        "meta_key": "meta_prolific",
        "out_subdir": "prolific-audio-emotion-prompt",
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
# AUDIO EMOTION PROMPTS
# =========================

# ---------- SIMPLE ----------
AUDIO_EMOTION_SIMPLE_INTRO = dedent("""
You are a senior expert in Speech Emotion Recognition (SER).

Your task is to analyze the given audio clip of a speaker and determine their emotional state.
""").strip()

AUDIO_EMOTION_SIMPLE_FORMAT = dedent("""
emotion: [predicted emotional state]
reasoning: Briefly explain the key acoustic cues supporting this emotion.
""").strip()

# ---------- FINE ----------
AUDIO_EMOTION_FINE_INTRO = dedent("""
You are a senior expert in Speech Emotion Recognition (SER).

Your task is to analyze fine-grained vocal characteristics that reflect the speaker’s emotional state.
""").strip()

AUDIO_EMOTION_FINE_FORMAT = dedent("""
Emotion:
Pitch:
Speech Rate:
Energy:
Intonation:
Overall Vocal Dynamics:
""").strip()

# ---------- REASONING (emotion ↔ personality) ----------
AUDIO_EMOTION_REASONING_HEAD = dedent("""
You are an expert in affective psychology and personality-oriented emotion analysis.

Your task is to analyze the emotional state expressed in response to a specific question
and reason about how this emotion relates to a particular personality dimension
activated by the question context.

If the question does not correspond to a specific personality dimension,
treat it as a general emotional context.
""").strip()

AUDIO_EMOTION_REASONING_FORMAT = dedent("""
Question Context:
- Activated Personality Dimension: {dimension}

emotion: [predicted emotional state]
reasoning: Based on the emotional expression observed in response to this question,
it is inferred that the emotion reflects how the individual’s affective response
supports, contradicts, or contextualizes the {dimension} dimension
(or represents a general emotional tendency if no specific dimension is activated).
""").strip()

# =========================
# HELPERS
# =========================
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def get_question_keys(meta: dict) -> List[str]:
    q_dict = meta.get("questions", {})
    if not isinstance(q_dict, dict) or len(q_dict) == 0:
        raise KeyError("meta['questions'] missing or empty")
    return sorted(q_dict.keys())


# =========================
# BUILD DATAFRAMES
# =========================
def build_simple_fine_df(
    transcripts: pd.DataFrame,
    meta: dict,
    style: str,
) -> pd.DataFrame:
    if style == "simple":
        prompt = AUDIO_EMOTION_SIMPLE_INTRO + "\n\n" + AUDIO_EMOTION_SIMPLE_FORMAT
    else:
        prompt = AUDIO_EMOTION_FINE_INTRO + "\n\n" + AUDIO_EMOTION_FINE_FORMAT

    col = f"{style}_prompt"
    qkeys = get_question_keys(meta)

    rows = []
    for _, row in transcripts.iterrows():
        pid = str(row["participantid"])
        for qk in qkeys:
            rows.append({
                "participantid": pid,
                "qkey": qk,
                col: prompt,
            })

    return pd.DataFrame(rows)


def build_reasoning_df(
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
            prompt_text = (
                AUDIO_EMOTION_REASONING_HEAD
                + "\n\n"
                + AUDIO_EMOTION_REASONING_FORMAT.format(dimension=dimension)
            )

            rows.append({
                "participantid": pid,
                "qkey": qk,
                "activated_dimension": dimension,
                "reasoning_prompt": prompt_text,
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

    if EMOTION_PROMPT_STYLE in ("simple", "fine"):
        df = build_simple_fine_df(
            transcripts=transcripts,
            meta=meta,
            style=EMOTION_PROMPT_STYLE,
        )
    elif EMOTION_PROMPT_STYLE == "reasoning":
        df = build_reasoning_df(
            transcripts=transcripts,
            meta=meta,
            dataset=SELECT_DATASET,
        )
    else:
        raise ValueError("Invalid EMOTION_PROMPT_STYLE")

    out_path = out_dir / f"questions_audio_emotion_{safe_name(EMOTION_PROMPT_STYLE)}_{safe_name(QUESTION_TYPE)}.pkl"
    save_df(df, out_path)

    print(
        f"[OK] Audio emotion prompts generated | "
        f"Dataset={SELECT_DATASET} | "
        f"Style={EMOTION_PROMPT_STYLE} | "
        f"Rows={len(df)} | "
        f"Saved={out_path}"
    )

if __name__ == "__main__":
    main()
