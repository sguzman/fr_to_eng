from transformers import MarianMTModel, MarianTokenizer
import os

# Load MarianMT model for German to English
model_name = "Helsinki-NLP/opus-mt-de-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Helper: Translate a single chunk
def translate_chunk(text):
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Read the full German text
def read_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Split into smaller chunks (~500 tokens worth)
def split_into_chunks(text, max_chars=1500):
    import textwrap
    return textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)

# Translate entire text file
def translate_file(input_path, output_path):
    print("Reading input...")
    german_text = read_text(input_path)
    chunks = split_into_chunks(german_text)

    print(f"Translating {len(chunks)} chunks...")
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i + 1} / {len(chunks)}")
        translated = translate_chunk(chunk)
        translated_chunks.append(translated)

    # Write to output
    print("Writing output...")
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write('\n\n'.join(translated_chunks))

    print(f"Done. Translated file saved to: {output_path}")

# === Usage ===
if __name__ == "__main__":
    input_txt = "german_input.txt"      # Replace with your file
    output_txt = "translated_output.txt"
    translate_file(input_txt, output_txt)

