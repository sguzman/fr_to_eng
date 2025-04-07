import logging
from transformers import MarianMTModel, MarianTokenizer
import textwrap

# === Setup logging ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("translation_debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# === Load MarianMT French → English ===
model_name = "Helsinki-NLP/opus-mt-fr-en"
logger.info(f"Loading model: {model_name}")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# === Translate one chunk ===
def translate_chunk(text: str) -> str:
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# === Split input ===
def split_into_chunks(text: str, max_chars: int = 1500) -> list[str]:
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
    input_txt = "french_input.txt"     # Change if needed
    output_txt = "translated_output.txt"
    translate_file(input_txt, output_txt)

