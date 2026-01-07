import os
import pandas as pd
from llm import Qwen2_5_LLM

# =========================
# USER SETTINGS
# =========================
MODEL_NAME = "qwen2.5"

DATASET = "prolific"   # "opva" / "prolific"


PROMPT_PKL = (
    "/root/autodl-tmp/data/personality_prompt/"
    "questions_personality_text_prolific.pkl"
)

OUTPUT_ROOT = (
    f"/root/autodl-tmp/output_data/{MODEL_NAME}/"
    f"qwen2.5+prolific_text_personality"
)

# =========================
# LOAD TEXT-ONLY MODEL
# =========================
llm = Qwen2_5_LLM(
    mode_name_or_path="/root/autodl-tmp/Qwen/Qwen2.5-7B-Instruct"
)

def get_answer(prompt_text: str) -> str:
    out = llm(prompt_text)
    return out if out is not None else ""

# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(PROMPT_PKL):
        raise FileNotFoundError(f"Prompt PKL not found: {PROMPT_PKL}")

    print(f"\n===== Running TEXT PERSONALITY inference: {DATASET} =====")

    df = pd.read_pickle(PROMPT_PKL)


    required_cols = {
        "participantid",
        "qkey",
        "activated_dimension",
        "personality_prompt",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Prompt PKL must contain columns {required_cols}, "
            f"but got {list(df.columns)}"
        )

    # 输出根目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for _, row in df.iterrows():
        pid = str(row["participantid"])
        qkey = str(row["qkey"])
        dimension = str(row["activated_dimension"])
        prompt_text = row["personality_prompt"]

        # 按人格维度建文件夹
        dim_dir = os.path.join(OUTPUT_ROOT, dimension)
        os.makedirs(dim_dir, exist_ok=True)

        out_path = os.path.join(dim_dir, f"{pid}_{qkey}.txt")
        if os.path.exists(out_path):
            continue

        try:
            print(f"[RUN] {pid}_{qkey} | {dimension}")
            result = get_answer(prompt_text)

            if result.strip():
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"[SAVE] {out_path}")
            else:
                print(f"[EMPTY] {pid}_{qkey}")

        except Exception as e:
            print(f"[ERROR] {pid}_{qkey}: {e}")

if __name__ == "__main__":
    main()
