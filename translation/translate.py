import json
import math
import torch
import re
from typing import Dict, Any, Optional
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------
# MODEL PRESETS
# ---------------
MODEL_PRESETS = {
    # M2M100
    "m2m100_418m": {
        "model_name": "facebook/m2m100_418M",
        "src_lang": "bn",
        "tgt_lang": "sw",
        "lang_key": "sw",
    },

    # NLLB-200 family
    "nllb200_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "src_lang": "ben_Beng",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
    "nllb200_1.3b": {
        "model_name": "facebook/nllb-200-1.3B",
        "src_lang": "ben_Beng",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
    "nllb200_3.3b": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "ben_Beng",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
}

# -----------------------------
# Sentence splitting
# -----------------------------
_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+|\n+")


def _split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in _SPLIT_RE.split(text.strip()) if s.strip()]


def _build_translator(model_name: str, src_lang: str, tgt_lang: str, device: int = 0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float16, 
    ).to(f"cuda:{device}")
    model.eval()


    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,   
    )

    return translator


split_prompt_count, split_response_count = 0, 0

def split_sentences_bn(text: str, is_prompt: bool = True):
    """Split text into sentences and update counters."""
    global split_prompt_count, split_response_count

    if not isinstance(text, str) or not text.strip():
        return []

    sentences = [s.strip() for s in _SPLIT_RE.split(text.strip()) if s.strip()]

    if len(sentences) > 1:
        if is_prompt:
            split_prompt_count += 1
        else:
            split_response_count += 1

    return sentences

def _nllb_forced_bos_id(tokenizer, tgt_lang: str) -> int:
    """
    Robustly map NLLB language code (e.g., 'hau_Latn') to the BOS token id
    across tokenizer variants / versions.
    """
    # 1) try vocab lookup
    vocab = tokenizer.get_vocab()
    if tgt_lang in vocab:
        return vocab[tgt_lang]

    # 2) try token->id conversion
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if bos_id is not None and bos_id != tokenizer.unk_token_id:
        return bos_id

    # 3) common alt: some tokenizers store as special token like "__hau_Latn__" (rare)
    alt = f"__{tgt_lang}__"
    if alt in vocab:
        return vocab[alt]
    bos_id = tokenizer.convert_tokens_to_ids(alt)
    if bos_id is not None and bos_id != tokenizer.unk_token_id:
        return bos_id

    raise KeyError(
        f"Could not find BOS id for tgt_lang='{tgt_lang}'. "
        f"Check you are using an NLLB tokenizer and the code is valid (e.g., eng_Latn, hau_Latn)."
    )


def batch_translate_texts_patched(
    translator,
    texts,
    *,
    src_lang: str,
    tgt_lang: str,
    is_prompt=True,
    sent_batch_size=32,
    max_new_tokens=256,
    truncation_max_length=256,
):
    batched_sentences, indices = [], []
    for i, text in enumerate(texts):
        for sent in split_sentences_bn(text, is_prompt=is_prompt):
            sent = sent.strip()
            if sent:
                batched_sentences.append(sent)
                indices.append(i)

    if not batched_sentences:
        return texts

    tokzr = translator.tokenizer
    model = translator.model
    device = model.device

    # ✅ NLLB needs src_lang and forced_bos_token_id
    if hasattr(tokzr, "src_lang"):
        tokzr.src_lang = src_lang

    forced_bos = _nllb_forced_bos_id(tokzr, tgt_lang)

    outputs = [""] * len(texts)

    for start in range(0, len(batched_sentences), sent_batch_size):
        chunk = batched_sentences[start : start + sent_batch_size]
        chunk_indices = indices[start : start + sent_batch_size]

        tok = tokzr(
            chunk,
            padding=True,
            truncation=True,
            max_length=truncation_max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        # with torch.no_grad():
        with torch.inference_mode():
            gen = model.generate(
                **tok,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        decoded = tokzr.batch_decode(gen, skip_special_tokens=True)

        for idx, t in zip(chunk_indices, decoded):
            t = (t or "").strip()
            if not t:
                continue
            outputs[idx] = (outputs[idx] + " " + t).strip() if outputs[idx] else t

    # fallback to original if empty
    for i in range(len(outputs)):
        if not outputs[i] and isinstance(texts[i], str):
            outputs[i] = texts[i]

    return outputs


def batch_translate_texts_patched_old(
    translator,
    texts,
    *,
    src_lang: str,
    tgt_lang: str,
    is_prompt=True,
    sent_batch_size=32,
    max_new_tokens=256,
    truncation_max_length=256,
):
    # split -> flatten
    batched_sentences, indices = [], []
    for i, text in enumerate(texts):
        for sent in split_sentences_bn(text, is_prompt=is_prompt):
            if sent.strip():
                batched_sentences.append(sent)
                indices.append(i)

    if not batched_sentences:
        return texts

    tokzr = translator.tokenizer
    model = translator.model
    device = model.device

    # ✅ NLLB requires these
    tokzr.src_lang = src_lang
    try:
        forced_bos = tokzr.lang_code_to_id[tgt_lang]
    except KeyError:
        raise KeyError(
            f"{tgt_lang} not in tokenizer.lang_code_to_id. "
            f"Double-check codes like eng_Latn, hau_Latn, ben_Beng."
        )

    decoded_all = []

    # ✅ actually use sent_batch_size
    for start in range(0, len(batched_sentences), sent_batch_size):
        chunk = batched_sentences[start : start + sent_batch_size]

        tok = tokzr(
            chunk,
            padding=True,
            truncation=True,
            max_length=truncation_max_length,  # input-side cap
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **tok,
                forced_bos_token_id=forced_bos,  # ✅ target language
                max_new_tokens=max_new_tokens,   # output-side cap
                num_beams=1,
                do_sample=False,
            )

        decoded_all.extend(tokzr.batch_decode(out, skip_special_tokens=True))

    # reconstruct per original text
    outputs = [""] * len(texts)
    for i, t in zip(indices, decoded_all):
        t = (t or "").strip()
        if t:
            outputs[i] = (outputs[i] + " " + t).strip() if outputs[i] else t

    # fallback to original if empty
    for i in range(len(outputs)):
        if not outputs[i] and isinstance(texts[i], str):
            outputs[i] = texts[i]

    return outputs

def _batch_translate_texts(
    translator,
    texts,
    *,
    is_prompt: bool,
    sent_batch_size: int,
    max_length: int,
    stats: Dict[str, int],
):
    """
    Sentence split -> flatten -> translate in sentence batches -> reconstruct per original text.
    """
    batched_sentences = []
    indices = []

    for idx, text in enumerate(texts):
        sents = _split_sentences(text)
        if len(sents) > 1:
            if is_prompt:
                stats["split_prompt_count"] += 1
            else:
                stats["split_response_count"] += 1

        for sent in sents:
            if sent.strip():
                batched_sentences.append(sent)
                indices.append(idx)

    if not batched_sentences:
        return texts

    translations = translator(
        batched_sentences,
        max_length=max_length,
        batch_size=sent_batch_size,
    )

    outputs = [""] * len(texts)
    for idx, trans in zip(indices, translations):
        t = trans.get("translation_text", "").strip()
        if not t:
            continue
        outputs[idx] = (outputs[idx] + " " + t).strip() if outputs[idx] else t

    # fallback: keep original if empty
    for i in range(len(outputs)):
        if not outputs[i] and isinstance(texts[i], str):
            outputs[i] = texts[i]

    return outputs


def translate(
    *,
    input_path: str,
    output_path: str,
    dataset: str = None,
    model_preset: str = "nllb200_600m",
    gpu_device: int = 0,
    obj_batch_size: int = 128,
    sent_batch_size: int = 32,
    max_new_tokens: int = 256,
    target_lang_key: Optional[str] = None,
    src_lang_override: Optional[str] = None,
    tgt_lang_override: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Translate fields `agg_prompt_bn` and `agg_response_bn` and inject:

      "translation": {
         "<lang_key>": {
             "prompt_translated_lang": "...",
             "prompt_back_to_original_lang": "...",
             "response_translated_lang": "...",
             "response_back_to_original_lang": "...",
             "model_name": "..."
         }
      }

    at the same level as agg_prompt_bn/agg_response_bn.

    Returns a dict of stats.
    """
    if model_preset not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model_preset={model_preset}. Available: {sorted(MODEL_PRESETS.keys())}"
        )

    preset = MODEL_PRESETS[model_preset]
    model_name = preset["model_name"]
    src_lang = src_lang_override or preset["src_lang"]
    tgt_lang = tgt_lang_override or preset["tgt_lang"]
    lang_key = target_lang_key or preset.get("lang_key", "sw")

    stats = {
        "model_preset": model_preset,
        "model_name": model_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "lang_key": lang_key,
        "gpu_device": gpu_device,
        "split_prompt_count": 0,
        "split_response_count": 0,
        "num_objects": 0,
    }

    translator = _build_translator(
        model_name=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=gpu_device,
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    stats["num_objects"] = len(data)

    num_batches = math.ceil(len(data) / obj_batch_size)

    iterator = range(num_batches)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Translating ({src_lang}->{tgt_lang})", unit="batch")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")

        for b in iterator:
            batch_objs = data[b * obj_batch_size : (b + 1) * obj_batch_size]

            

            if dataset == "Aegis":
                prompts = [(obj.get("prompt") or "").strip() for obj in batch_objs]
                responses = [(obj.get("response") or "").strip() for obj in batch_objs]
            else:
                prompts = [(obj.get("agg_prompt_bn") or "").strip() for obj in batch_objs]
                responses = [(obj.get("agg_response_bn") or "").strip() for obj in batch_objs]

            prompts_tr = batch_translate_texts_patched(
                translator,
                prompts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                sent_batch_size=sent_batch_size,
                max_new_tokens=max_new_tokens,
                truncation_max_length=256,
            )

            responses_tr = batch_translate_texts_patched(
                translator,
                responses,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                sent_batch_size=sent_batch_size,
                max_new_tokens=max_new_tokens,
                truncation_max_length=256,
                is_prompt=False,
            )

            prompts_back = batch_translate_texts_patched(
                translator,
                prompts_tr,
                src_lang=tgt_lang,
                tgt_lang=src_lang,
                sent_batch_size=sent_batch_size,
                max_new_tokens=max_new_tokens,
                truncation_max_length=256,
            )

            responses_back = batch_translate_texts_patched(
                translator,
                responses_tr,
                src_lang=tgt_lang,
                tgt_lang=src_lang,
                sent_batch_size=sent_batch_size,
                max_new_tokens=max_new_tokens,
                truncation_max_length=256,
                is_prompt=False,
            )

            for obj, p_t, p_b, r_t, r_b in zip(batch_objs, prompts_tr, prompts_back, responses_tr, responses_back):
                if "translation" not in obj or not isinstance(obj["translation"], dict):
                    obj["translation"] = {}

                obj["translation"][lang_key] = {
                    "prompt_translated_lang": p_t,
                    "prompt_back_to_original_lang": p_b,
                    "response_translated_lang": r_t,
                    "response_back_to_original_lang": r_b,
                    "model_name": model_name,
                }

            for i, obj in enumerate(batch_objs):
                json.dump(obj, f_out, ensure_ascii=False, indent=2)
                is_last = (b == num_batches - 1) and (i == len(batch_objs) - 1)
                f_out.write("\n" if is_last else ",\n")

            f_out.flush()

        f_out.write("]\n")

    stats["output_path"] = output_path
    return stats


def translate_languages(
    *,
    input_path: str,
    output_path: str,
    languages: list,
    lang_name_to_code: Dict[str, str],
    src_lang_code: str = "eng_Latn",
    model_name: str = "facebook/nllb-200-3.3B",
    gpu_device: int = 0,
    batch_size: int = 64,
    sent_batch_size: int = 32,
    max_new_tokens: int = 256,
    save_every_language: bool = True,
    dataset: str = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Translate prompt/response fields into multiple target languages with back-translation.

    For each language in `languages`, injects under each row's "translation" key:

      "<lang_name>": {
          "prompt_translated_lang": "...",
          "prompt_back_to_original_lang": "...",
          "response_translated_lang": "...",
          "response_back_to_original_lang": "...",
          "model_name": "..."
      }

    Supports resuming: if output_path already exists, rows are loaded from there so
    any previously completed language translations are preserved.

    Args:
        input_path: Path to the source JSON file (list of objects).
        output_path: Path to write the output JSON file.
        languages: List of language names to translate into (e.g. ["Hausa", "Swahili"]).
        lang_name_to_code: Mapping from language name to NLLB language code
                           (e.g. {"Hausa": "hau_Latn", "Swahili": "swh_Latn"}).
        src_lang_code: NLLB source language code (default: "eng_Latn").
        model_name: HuggingFace model name (default: facebook/nllb-200-3.3B).
        gpu_device: CUDA device index.
        batch_size: Number of rows to process per object batch.
        sent_batch_size: Number of sentences per translation batch.
        max_new_tokens: Max tokens for generation.
        save_every_language: If True, saves output after each language completes.
        dataset: "Aegis" to read prompt/response fields; otherwise reads agg_prompt_bn/agg_response_bn.
        show_progress: Whether to show tqdm progress bars.

    Returns:
        Dict with stats (languages_completed, languages_skipped, num_rows, total_seconds).
    """
    import os
    import time

    # --- load rows (resume from output if it exists) ---
    load_path = output_path if os.path.exists(output_path) else input_path
    with open(load_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {load_path}.")

    for row in rows:
        if "translation" not in row or not isinstance(row["translation"], dict):
            row["translation"] = {}

    # --- build translator once (NLLB handles all language pairs via src/tgt override) ---
    translator = _build_translator(
        model_name=model_name,
        src_lang=src_lang_code,
        tgt_lang=src_lang_code,  # placeholder; overridden per call
        device=gpu_device,
    )

    total_rows = len(rows)
    languages_completed = []
    languages_skipped = []
    start_all = time.time()

    def _translate_in_batches(texts, src, tgt):
        results = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            translated = batch_translate_texts_patched(
                translator,
                chunk,
                src_lang=src,
                tgt_lang=tgt,
                sent_batch_size=sent_batch_size,
                max_new_tokens=max_new_tokens,
                truncation_max_length=256,
            )
            results.extend(translated)
        return results

    lang_iter = enumerate(languages, start=1)
    if show_progress:
        lang_iter = tqdm(list(lang_iter), desc="Languages", unit="lang")

    for idx, lang_name in lang_iter:

        # ── Skip check: count how many rows already have this language ──
        done_count = sum(1 for row in rows if lang_name in row.get("translation", {}))
        if done_count == total_rows:
            languages_skipped.append(lang_name)
            if show_progress:
                tqdm.write(f"  [{idx}/{len(languages)}] {lang_name} SKIPPED (already complete)")
            continue
        elif done_count > 0 and show_progress:
            tqdm.write(f"  [{idx}/{len(languages)}] {lang_name} partial ({done_count}/{total_rows}), resuming...")

        # ── Validate language code before doing any work ──
        tgt_code = lang_name_to_code.get(lang_name)
        if not tgt_code:
            languages_skipped.append(lang_name)
            if show_progress:
                tqdm.write(f"  [{idx}/{len(languages)}] {lang_name} SKIPPED (no NLLB code — unsupported)")
            continue

        # ── Validate the code is actually in the tokenizer vocab ──
        tokzr = translator.tokenizer
        vocab = tokzr.get_vocab()
        if tgt_code not in vocab and tokzr.convert_tokens_to_ids(tgt_code) == tokzr.unk_token_id:
            languages_skipped.append(lang_name)
            if show_progress:
                tqdm.write(f"  [{idx}/{len(languages)}] {lang_name} SKIPPED (invalid NLLB code '{tgt_code}')")
            continue

        lang_start = time.time()

        if dataset == "Aegis":
            prompts   = [(row.get("prompt")    or "").strip() for row in rows]
            responses = [(row.get("response")  or "").strip() for row in rows]
        else:
            prompts   = [(row.get("agg_prompt_bn")   or "").strip() for row in rows]
            responses = [(row.get("agg_response_bn") or "").strip() for row in rows]

        try:
            prompts_tr     = _translate_in_batches(prompts,      src_lang_code, tgt_code)
            responses_tr   = _translate_in_batches(responses,    src_lang_code, tgt_code)
            prompts_back   = _translate_in_batches(prompts_tr,   tgt_code,      src_lang_code)
            responses_back = _translate_in_batches(responses_tr, tgt_code,      src_lang_code)

            for i, row in enumerate(rows):
                row["translation"][lang_name] = {
                    "prompt_translated_lang":      prompts_tr[i],
                    "prompt_back_to_original_lang": prompts_back[i],
                    "response_translated_lang":    responses_tr[i],
                    "response_back_to_original_lang": responses_back[i],
                    "model_name": model_name,
                }

            languages_completed.append(lang_name)

            if show_progress:
                elapsed = time.time() - lang_start
                tqdm.write(f"  [{idx}/{len(languages)}] {lang_name} done in {elapsed:.1f}s")

            if save_every_language:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)

        except KeyError as e:
            # Bad language code — log and skip, don't crash
            languages_skipped.append(lang_name)
            if show_progress:
                tqdm.write(f"  [{idx}/{len(languages)}] {lang_name} SKIPPED — KeyError: {e}")
            if save_every_language:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)

        except RuntimeError as e:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"CUDA OOM on language '{lang_name}'. Progress saved to {output_path}. "
                    f"Try reducing batch_size (current: {batch_size})."
                ) from e
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return {
        "num_rows": total_rows,
        "languages_completed": languages_completed,
        "languages_skipped": languages_skipped,
        "output_path": output_path,
        "total_seconds": time.time() - start_all,
    }


if __name__ == "__main__":
    LANG_NAME_TO_CODE = {
        # High-resource
        "Arabic":                   "arb_Arab",
        "Chinese (Simplified)":     "zho_Hans",
        "Chinese (Traditional)":    "zho_Hant",
        "English":                  "eng_Latn",
        "French":                   "fra_Latn",
        "German":                   "deu_Latn",
        "Italian":                  "ita_Latn",
        "Japanese":                 "jpn_Jpan",
        "Korean":                   "kor_Hang",
        "Portuguese":               "por_Latn",
        "Russian":                  "rus_Cyrl",
        "Spanish":                  "spa_Latn",
        # Medium-resource
        "Bengali":                  "ben_Beng",
        "Bulgarian":                "bul_Cyrl",
        "Czech":                    "ces_Latn",
        "Danish":                   "dan_Latn",
        "Dutch":                    "nld_Latn",
        "Finnish":                  "fin_Latn",
        "Greek":                    "ell_Grek",
        "Hebrew":                   "heb_Hebr",
        "Hindi":                    "hin_Deva",
        "Indonesian":               "ind_Latn",
        "Malay":                    "zsm_Latn",
        "Norwegian":                "nob_Latn",
        "Persian":                  "pes_Arab",
        "Polish":                   "pol_Latn",
        "Romanian":                 "ron_Latn",
        "Swedish":                  "swe_Latn",
        "Thai":                     "tha_Thai",
        "Turkish":                  "tur_Latn",
        "Ukrainian":                "ukr_Cyrl",
        "Urdu":                     "urd_Arab",
        "Vietnamese":               "vie_Latn",
        # Low-resource
        "Afrikaans":                "afr_Latn",
        "Amharic":                  "amh_Ethi",
        "Armenian":                 "hye_Armn",
        "Assamese":                 "asm_Beng",
        "Asturian":                 "ast_Latn",
        "Azerbaijani":              "azj_Latn",
        "Belarusian":               "bel_Cyrl",
        "Bosnian":                  "bos_Latn",
        "Burmese":                  "mya_Mymr",
        "Catalan":                  "cat_Latn",
        "Cebuano":                  "ceb_Latn",
        "Croatian":                 "hrv_Latn",
        "Estonian":                 "est_Latn",
        "Filipino (Tagalog)":       "tgl_Latn",
        "Fula":                     "fuv_Latn", #fub_Arab
        "Galician":                 "glg_Latn",
        "Ganda":                    "lug_Latn",
        "Georgian":                 "kat_Geor",
        "Gujarati":                 "guj_Gujr",
        "Hausa":                    "hau_Latn",
        "Hungarian":                "hun_Latn",
        "Icelandic":                "isl_Latn",
        "Igbo":                     "ibo_Latn",
        "Irish":                    "gle_Latn",
        "Javanese":                 "jav_Latn",
        "Kabuverdianu":             None, #"kea_Latn" not supported by NLLB-200-3.3B
        "Kamba":                    None, #"kam_Latn", not supported
        "Kannada":                  "kan_Knda",
        "Kazakh":                   "kaz_Cyrl",
        "Khmer":                    "khm_Khmr",
        "Kyrgyz":                   "kir_Cyrl",
        "Lao":                      "lao_Laoo",
        "Latvian":                  "lvs_Latn",
        "Lingala":                  "lin_Latn",
        "Lithuanian":               "lit_Latn",
        "Luo":                      "luo_Latn",
        "Luxembourgish":            "ltz_Latn",
        "Macedonian":               "mkd_Cyrl",
        "Malayalam":                "mal_Mlym",
        "Maltese":                  "mlt_Latn",
        "Maori":                    "mri_Latn",
        "Marathi":                  "mar_Deva",
        "Mongolian":                "khk_Cyrl",
        "Nepali":                   "npi_Deva",
        "Northern Sotho":           "nso_Latn",
        "Nyanja":                   "nya_Latn",
        "Occitan":                  "oci_Latn",
        "Oriya":                    "ory_Latn", #ory_Orya
        "Oriya (Odia)":             "ory_Orya",
        "Oromo":                    "gaz_Latn",
        "Pashto":                   "pbt_Arab",
        "Punjabi":                  "pan_Guru",
        "Serbian":                  "srp_Cyrl",
        "Shona":                    "sna_Latn",
        "Sindhi":                   "snd_Arab",
        "Slovak":                   "slk_Latn",
        "Slovenian":                "slv_Latn",
        "Somali":                   "som_Latn",
        "Sorani Kurdish":           "ckb_Arab",
        "Swahili":                  "swh_Latn",
        "Tajik":                    "tgk_Cyrl",
        "Tamil":                    "tam_Taml",
        "Telugu":                   "tel_Telu",
        "Umbundu":                  None, # "umb_Latn", not supported
        "Welsh":                    "cym_Latn",
        "Wolof":                    "wol_Latn",
        "Xhosa":                    "xho_Latn",
        "Yoruba":                   "yor_Latn",
        "Zulu":                     "zul_Latn",
    }

    TARGET_LANGUAGES = list(LANG_NAME_TO_CODE.keys())

    translation_output_base = f"{dataset_base_path}/translation"
    aegis_dataset_filepath = "/mmse-bench-anonym/dataset-sourcing/data/aegis.json"
    translation_output_filepath = "/mmse-bench-anonym/translation/data/aegis_multilang.json"

    stats = translate_languages(
            input_path=aegis_dataset_path,
            output_path=translation_output_filepath,
            languages=TARGET_LANGUAGES,
            lang_name_to_code=LANG_NAME_TO_CODE,
            src_lang_code="eng_Latn",
            model_name="facebook/nllb-200-3.3B",
            gpu_device=0,
            batch_size=64,
            sent_batch_size=32,
            save_every_language=True,
            dataset="Aegis",
        )
    print(stats)

