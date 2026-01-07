import os
import pickle
import torch
import pandas as pd
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =========================
# USER SETTINGS
# =========================
HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

DATASET_TO_RUN = "opva"      #只跑一个数据库："opva" 或 "prolific"

PROMPT_PKLS = {
    "opva": r"D:\code\code-Experiment1\opva-emotion-prompt\questions_emotion_reasoning_factors.pkl",
    "prolific": r"D:\code\code-Experiment1\prolific-emotion-prompt\questions_emotion_reasoning_factors.pkl",
}

VIDEO_DIR = r"D:\code\code-Experiment1\data\video"  # 换成自己的视频数据路径
OUTPUT_ROOT = r"D:\code\code-Experiment1\output_reasoning"

MAX_PIXELS = 360 * 420
FPS = 1.0
MAX_NEW_TOKENS = 4096

# =========================
# MODEL (load once)
# =========================
_model = None
_processor = None


def load_model_and_processor():
    global _model, _processor
    if _model is not None:
        return

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    _model.eval()
    _processor = AutoProcessor.from_pretrained(HF_MODEL_ID)


def get_device():
    if torch.cuda.is_available():
        return next(_model.parameters()).device
    return torch.device("cpu")


@torch.no_grad()
def run_video_inference(video_path: str, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": MAX_PIXELS,
                    "fps": FPS,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    inputs = inputs.to(get_device())

    outputs = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    outputs = outputs[:, inputs.input_ids.shape[1]:]

    decoded = _processor.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return decoded[0] if decoded else ""


# =========================
# MAIN PIPELINE
# =========================
def main():
    load_model_and_processor()

    if DATASET_TO_RUN not in PROMPT_PKLS:
        raise ValueError(f"Invalid DATASET_TO_RUN: {DATASET_TO_RUN}")

    prompt_pkl_path = PROMPT_PKLS[DATASET_TO_RUN]
    if not os.path.exists(prompt_pkl_path):
        raise FileNotFoundError(f"Prompt PKL not found: {prompt_pkl_path}")

    print(f"\n===== Running dataset: {DATASET_TO_RUN} =====")

    df = pd.read_pickle(prompt_pkl_path)

    required_cols = {"participantid", "qkey", "reasoning_prompt"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Prompt PKL must contain columns {required_cols}, "
            f"but got {list(df.columns)}"
        )

    out_dir = os.path.join(OUTPUT_ROOT, DATASET_TO_RUN)
    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        pid = str(row["participantid"])
        qkey = str(row["qkey"])
        prompt_text = row["reasoning_prompt"]

        video_name = f"{pid}_{qkey}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_name)

        # 判断视频是否存在
        if not os.path.exists(video_path):
            continue

        out_path = os.path.join(out_dir, f"{pid}_{qkey}.txt")
        if os.path.exists(out_path):
            continue

        try:
            print(f"[RUN] {DATASET_TO_RUN} | {video_name}")
            result = run_video_inference(video_path, prompt_text)

            if result.strip():
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"[SAVE] {out_path}")
            else:
                print(f"[EMPTY] {video_name}")

        except Exception as e:
            print(f"[ERROR] {video_name}: {e}")


if __name__ == "__main__":
    main()
