import logging
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

# === GPU support ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# === Load NLLB-200 model ===
model_name = "facebook/nllb-200-distilled-600M"
from transformers import NllbTokenizer
tokenizer = NllbTokenizer.from_pretrained(model_name)
logger.debug(f"Available language codes: {tokenizer.lang_code_to_id.keys()}")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

source_lang = "fra_Latn"
target_lang = "eng_Latn"

# === Translate one chunk ===
def translate_chunk(text: str) -> str:
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    tokens["forced_bos_token_id"] = tokenizer.lang_code_to_id[target_lang]
    translated_tokens = model.generate(**tokens)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# === Chunk text ===
def split_into_chunks(text: str, max_chars: int = 1000) -> list[str]:
    return textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)

# === File I/O ===
def read_text(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_text(filepath: str, text: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

# === Main translation pipeline ===
def translate_file(input_path: str, output_path: str):
    logger.info(f"Reading input from {input_path}")
    french_text = read_text(input_path)

    logger.info("Splitting into chunks...")
    chunks = split_into_chunks(french_text)

    logger.info(f"{len(chunks)} chunks created. Beginning translation...")

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"\n--- Chunk {i + 1} / {len(chunks)} ---")
        logger.debug(f"[FRENCH] {chunk}\n")

        try:
            translation = translate_chunk(chunk)
        except Exception as e:
            logger.error(f"Error translating chunk {i+1}: {e}")
            translation = "[ERROR IN TRANSLATION]"

        logger.debug(f"[ENGLISH] {translation}\n")
        translated_chunks.append(translation)

    final_output = "\n\n".join(translated_chunks)

    logger.info(f"Writing translated output to {output_path}")
    write_text(output_path, final_output)

    logger.info("✅ Translation complete.")

# === Entrypoint ===
if __name__ == "__main__":
    input_txt = "french_input.txt"     # Customize as needed
    output_txt = "translated_output.txt"
    translate_file(input_txt, output_txt)

