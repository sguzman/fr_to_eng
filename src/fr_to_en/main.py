import logging
import textwrap
import torch
import sys
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# === Logging config ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("translation_debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Enforce GPU ===
if not torch.cuda.is_available():
    logger.error("❌ No GPU found. This script requires a GPU for translation.")
    sys.exit(1)

device = torch.device("cuda")
logger.info(f"✅ Using GPU device: {torch.cuda.get_device_name(0)}")

# === Load NLLB-200 model and tokenizer ===
model_name = "facebook/nllb-200-3.3B"
tokenizer = NllbTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=False).to(device)

# === Language codes ===
source_lang = "fra_Latn"
target_lang = "eng_Latn"

# === Translate a single chunk ===
def translate_chunk(text: str) -> str:
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    tokens["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(target_lang)
    translated_tokens = model.generate(**tokens)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# === Split text into manageable chunks ===
def split_into_chunks(text: str, max_chars: int = 1000) -> list[str]:
    return textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)

# === File I/O ===
def read_text(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_text(filepath: str, text: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

# === Main translation function ===
def translate_file(input_path: str, output_path: str):
    logger.info(f"📖 Reading input from: {input_path}")
    french_text = read_text(input_path)

    logger.info("✂️ Splitting text into chunks...")
    chunks = split_into_chunks(french_text)

    logger.info(f"🔁 Translating {len(chunks)} chunks...")

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"\n--- Chunk {i + 1} / {len(chunks)} ---")
        logger.debug(f"[FRENCH]\n{chunk}\n")

        try:
            translation = translate_chunk(chunk)
        except Exception as e:
            logger.error(f"⚠️ Error translating chunk {i+1}: {e}")
            translation = "[ERROR IN TRANSLATION]"

        logger.debug(f"[ENGLISH]\n{translation}\n")
        translated_chunks.append(translation)

    final_output = "\n\n".join(translated_chunks)

    logger.info(f"💾 Writing translated output to: {output_path}")
    write_text(output_path, final_output)

    logger.info("✅ Translation complete.")

# === Entrypoint ===
if __name__ == "__main__":
    input_txt = "french_input.txt"        # Replace with your file path
    output_txt = "translated_output.txt"  # Output file
    translate_file(input_txt, output_txt)

