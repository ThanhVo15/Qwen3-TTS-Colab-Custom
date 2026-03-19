# %cd /content/Qwen3-TTS-Colab
import csv
import gc
import json
import os
import re
import subprocess
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from hf_downloader import download_model
from huggingface_hub import login
from huggingface_hub import snapshot_download
from process_text import text_chunk
from pydub import AudioSegment
from pydub.silence import split_on_silence
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from subtitle import subtitle_maker

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
  HF_TOKEN=None

# Global model holders
loaded_models = {}
MODEL_SIZES = ["0.6B", "1.7B"]

# Speaker and language choices
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]
BATCH_TABLE_HEADERS = ["Index", "Script", "Filename Preview", "Status"]
BATCH_MANIFEST_HEADERS = [
    "timestamp",
    "row_index",
    "filename",
    "status",
    "language",
    "chunks",
    "output_path",
    "message",
]
BATCH_SESSION = {
    "rows": [],
    "source_file": None,
    "prompt_items": None,
    "prompt_meta": None,
    "cancel_requested": False,
    "running": False,
}


class BatchCancelled(Exception):
    """Raised when the batch runner should stop after a safe checkpoint."""

# --- Helper Functions ---

def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    try:
      return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    except Exception as e:
      return download_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}", download_folder="./qwen_tts_model", redownload= False)

def clear_other_models(keep_key=None):
    """Delete all loaded models except the current one."""
    global loaded_models
    keys_to_delete = [k for k in loaded_models if k != keep_key]
    for k in keys_to_delete:
        try:
            del loaded_models[k]
        except Exception:
            pass
    for k in keys_to_delete:
        loaded_models.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_type: str, model_size: str):
    """Load model and clear others to avoid OOM in Colab."""
    global loaded_models
    key = (model_type, model_size)
    if key in loaded_models:
        return loaded_models[key]
    
    clear_other_models(keep_key=key)
    model_path = get_model_path(model_type, model_size)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    loaded_models[key] = model
    return model

def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None: return None
    if isinstance(audio, str):
        try:
            wav, sr = sf.read(audio)
            wav = _normalize_audio(wav)
            return wav, int(sr)
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None

def transcribe_reference(audio_path, mode_input, language="English"):
    """Uses subtitle_maker to extract text from the reference audio."""
    should_run = False
    if isinstance(mode_input, bool): should_run = mode_input
    elif isinstance(mode_input, str) and "High-Quality" in mode_input: should_run = True

    if not audio_path or not should_run: return gr.update()
    
    print(f"Starting transcription for: {audio_path}")
    src_lang = language if language != "Auto" else "English"
    try:
        results = subtitle_maker(audio_path, src_lang)
        transcript = results[7]
        return transcript if transcript else "Could not detect speech."
    except Exception as e:
        print(f"Transcription Error: {e}")
        return f"Error during transcription: {str(e)}"

# --- Audio Processing Utils (Disk Based) ---

def remove_silence_function(file_path, minimum_silence=100):
    """Removes silence from an audio file using Pydub."""
    try:
        output_path = file_path.replace(".wav", "_no_silence.wav")
        sound = AudioSegment.from_wav(file_path)
        audio_chunks = split_on_silence(sound,
                                        min_silence_len=minimum_silence,
                                        silence_thresh=-45,
                                        keep_silence=50)
        combined = AudioSegment.empty()
        for chunk in audio_chunks:
            combined += chunk
        combined.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error removing silence: {e}")
        return file_path

def process_audio_output(audio_path, make_subtitle, remove_silence, language="Auto"):
    """Handles Silence Removal and Subtitle Generation."""
    # 1. Remove Silence
    final_audio_path = audio_path
    if remove_silence:
        final_audio_path = remove_silence_function(audio_path)
    
    # 2. Generate Subtitles
    default_srt, custom_srt, word_srt, shorts_srt = None, None, None, None
    if make_subtitle:
        try:
            results = subtitle_maker(final_audio_path, language)
            default_srt = results[0]
            custom_srt = results[1]
            word_srt = results[2]
            shorts_srt = results[3]
        except Exception as e:
            print(f"Subtitle generation error: {e}")

    return final_audio_path, default_srt, custom_srt, word_srt, shorts_srt

def stitch_chunk_files(chunk_files,output_filename):
    """
    Takes a list of file paths.
    Stitches them into one file.
    Deletes the temporary chunk files.
    """
    if not chunk_files:
        return None

    combined_audio = AudioSegment.empty()
    
    print(f"Stitching {len(chunk_files)} audio files...")
    for f in chunk_files:
        try:
            segment = AudioSegment.from_wav(f)
            combined_audio += segment
        except Exception as e:
            print(f"Error appending chunk {f}: {e}")

    # output_filename = f"final_output_{os.getpid()}.wav"
    combined_audio.export(output_filename, format="wav")
    
    # Clean up temp files
    for f in chunk_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"Warning: Could not delete temp file {f}: {e}")
            
    return output_filename

# --- Generators (Memory Optimized) ---

def generate_voice_design(text, language, voice_description, remove_silence, make_subs):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None
    
    try:
        # 1. Chunk Text
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        print(f"Processing {len(text_chunks)} chunks...")
        
        chunk_files = []
        tts = get_model("VoiceDesign", "1.7B")

        # 2. Generate & Save Loop
        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_voice_design(
                text=chunk.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            
            # Save immediately to disk
            temp_filename = f"temp_chunk_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            
            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()
        
        # 3. Stitch from disk
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        
        # 4. Post-Process
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        
        return final_audio, "Generation Success!", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None

def generate_custom_voice(text, language, speaker, instruct, model_size, remove_silence, make_subs):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None
    
    try:
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        chunk_files = []
        tts = get_model("CustomVoice", model_size)
        formatted_speaker = speaker.lower().replace(" ", "_")

        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_custom_voice(
                text=chunk.strip(),
                language=language,
                speaker=formatted_speaker,
                instruct=instruct.strip() if instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            # Save immediately
            temp_filename = f"temp_custom_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            
            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()
            
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, "Generation Success!", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None

def smart_generate_clone(ref_audio, ref_text, target_text, language, mode, model_size, remove_silence, make_subs):
    if not target_text or not target_text.strip(): return None, "Error: Target text is required.", None, None, None, None
    if not ref_audio: return None, "Error: Ref audio required.", None, None, None, None

    # 1. Mode & Transcript Logic
    use_xvector_only = ("Fast" in mode)
    final_ref_text = ref_text
    audio_tuple = _audio_to_tuple(ref_audio)

    if not use_xvector_only:
        if not final_ref_text or not final_ref_text.strip():
            print("Auto-transcribing reference...")
            try:
                final_ref_text = transcribe_reference(ref_audio, True, language)
                if not final_ref_text or "Error" in final_ref_text:
                     return None, f"Transcription failed: {final_ref_text}", None, None, None, None
            except Exception as e:
                return None, f"Transcribe Error: {e}", None, None, None, None
    else:
        final_ref_text = None

    try:
        # 2. Chunk Target Text
        text_chunks, tts_filename = text_chunk(target_text, language, char_limit=280)
        chunk_files = []
        tts = get_model("Base", model_size)

        # 3. Generate Loop
        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_voice_clone(
                text=chunk.strip(),
                language=language,
                ref_audio=audio_tuple,
                ref_text=final_ref_text.strip() if final_ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            # Save immediately
            temp_filename = f"temp_clone_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)

            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()

        # 4. Stitch & Process
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, f"Success! Mode: {mode}", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None


# --- Batch Voice Clone Studio Helpers ---

def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_int(value, default):
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _clean_script_text(value) -> str:
    if _is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_language_choice(value, fallback="Auto") -> str:
    if _is_missing(value):
        return fallback
    normalized = str(value).strip()
    lookup = {item.lower(): item for item in LANGUAGES}
    return lookup.get(normalized.lower(), fallback)


def _normalize_enabled(value) -> bool:
    if _is_missing(value):
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no", "n", "off", "disabled"}:
        return False
    return True


def _make_safe_filename(raw_value, row_index, total_rows, used_names) -> str:
    width = max(3, len(str(max(total_rows, 1))))
    default_stem = f"{row_index:0{width}d}_audio"

    if _is_missing(raw_value):
        stem = default_stem
    else:
        stem = os.path.splitext(os.path.basename(str(raw_value).strip()))[0]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
        stem = stem or default_stem

    candidate = f"{stem}.wav"
    suffix = 2
    while candidate.lower() in used_names:
        candidate = f"{stem}_{suffix}.wav"
        suffix += 1

    used_names.add(candidate.lower())
    return candidate


def _clone_row_template(row_index: int) -> Dict[str, Any]:
    return {
        "row_index": row_index,
        "script": "",
        "filename": "",
        "status": "Pending",
        "language": None,
        "enabled": True,
        "output_path": None,
        "error": None,
        "chunks": 0,
        "note": None,
    }


def _row_is_final(row: Dict[str, Any]) -> bool:
    status = str(row.get("status", ""))
    return status.startswith("Success") or status.startswith("Failed") or status.startswith("Skipped")


def _rows_to_table_value(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    return [
        [row["row_index"], row["script"], row["filename"], row["status"]]
        for row in rows
    ]


def _table_value_to_rows(table_value, existing_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if table_value is None:
        return [dict(row) for row in existing_rows]

    raw_rows = table_value.values.tolist() if isinstance(table_value, pd.DataFrame) else list(table_value)
    existing_map = {int(row["row_index"]): dict(row) for row in existing_rows}
    normalized_rows = []
    seen_indices = set()

    for position, raw_row in enumerate(raw_rows, start=1):
        cells = list(raw_row) if isinstance(raw_row, (list, tuple)) else [raw_row]
        while len(cells) < len(BATCH_TABLE_HEADERS):
            cells.append("")

        requested_index = _safe_int(cells[0], position)
        if requested_index in seen_indices:
            requested_index = max(seen_indices) + 1
        seen_indices.add(requested_index)

        base = existing_map.get(requested_index, _clone_row_template(requested_index))
        script = _clean_script_text(cells[1] if len(cells) > 1 else base.get("script", ""))
        requested_filename = cells[2] if len(cells) > 2 else base.get("filename", "")

        row = dict(base)
        row["row_index"] = requested_index
        row["script"] = script
        row["_requested_filename"] = requested_filename

        if script != base.get("script", "") or str(requested_filename).strip() != str(base.get("filename", "")).strip():
            row["status"] = "Pending"
            row["output_path"] = None
            row["error"] = None
            row["chunks"] = 0
            row["note"] = None

        normalized_rows.append(row)

    used_names = set()
    total_rows = len(normalized_rows)
    for row in normalized_rows:
        row["filename"] = _make_safe_filename(row.pop("_requested_filename", row.get("filename")), row["row_index"], total_rows, used_names)
        if not row.get("enabled", True):
            row["status"] = "Skipped (disabled)"
        elif not row.get("script"):
            row["status"] = "Skipped (empty script)"

    return normalized_rows


def _reset_rows_for_new_run(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reset_rows = []
    for row in rows:
        updated = dict(row)
        updated["output_path"] = None
        updated["error"] = None
        updated["chunks"] = 0
        updated["note"] = None
        if not updated.get("enabled", True):
            updated["status"] = "Skipped (disabled)"
        elif not updated.get("script"):
            updated["status"] = "Skipped (empty script)"
        else:
            updated["status"] = "Pending"
        reset_rows.append(updated)
    return reset_rows


def _sync_batch_rows(table_value) -> List[Dict[str, Any]]:
    rows = _table_value_to_rows(table_value, BATCH_SESSION["rows"])
    BATCH_SESSION["rows"] = [dict(row) for row in rows]
    return rows


def _parse_selection_text(selection_text: str, valid_indices: List[int]) -> List[int]:
    if not selection_text or not selection_text.strip():
        raise ValueError("Enter one or more row indexes, for example: 1, 3-5")

    valid_set = set(valid_indices)
    selected = set()

    for part in selection_text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = _safe_int(start_text, None)
            end = _safe_int(end_text, None)
            if start is None or end is None:
                raise ValueError(f"Invalid row range: '{token}'")
            if start > end:
                start, end = end, start
            for item in range(start, end + 1):
                if item in valid_set:
                    selected.add(item)
        else:
            item = _safe_int(token, None)
            if item is None:
                raise ValueError(f"Invalid row index: '{token}'")
            if item in valid_set:
                selected.add(item)

    if not selected:
        raise ValueError("None of the selected row indexes exist in the current table.")

    return [item for item in valid_indices if item in selected]


def _normalize_output_folder_path(path_value: str) -> str:
    if not path_value or not path_value.strip():
        raise ValueError("Output folder path is required.")

    output_folder = os.path.expanduser(path_value.strip())
    if not os.path.isabs(output_folder):
        output_folder = os.path.abspath(output_folder)

    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def _get_batch_paths(output_folder: str) -> Dict[str, str]:
    return {
        "state": os.path.join(output_folder, "state.json"),
        "manifest": os.path.join(output_folder, "manifest.csv"),
        "prompt": os.path.join(output_folder, "voice_prompt.pt"),
    }


def _prompt_fingerprint(ref_audio_path: str, ref_text: str, model_size: str) -> str:
    try:
        stat_info = os.stat(ref_audio_path)
        audio_part = f"{os.path.abspath(ref_audio_path)}::{stat_info.st_size}::{int(stat_info.st_mtime)}"
    except OSError:
        audio_part = os.path.abspath(ref_audio_path) if ref_audio_path else "unknown_audio"

    return json.dumps(
        {
            "audio": audio_part,
            "text": _clean_script_text(ref_text),
            "model_size": model_size,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _format_prompt_status(prompt_meta: Optional[Dict[str, Any]], prompt_persisted: bool = False) -> str:
    if not prompt_meta:
        return "Prompt status: not cached"

    persisted_label = "saved to checkpoint" if prompt_persisted else "memory only"
    reference_name = prompt_meta.get("reference_name", "reference audio")
    model_size = prompt_meta.get("model_size", "unknown")
    return f"Prompt cached for Base {model_size} using {reference_name} ({persisted_label})."


def _serialize_prompt_items(prompt_items: List[VoiceClonePromptItem]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in prompt_items]


def _deserialize_prompt_items(items_raw: List[Dict[str, Any]]) -> List[VoiceClonePromptItem]:
    items: List[VoiceClonePromptItem] = []
    for payload in items_raw:
        ref_code = payload.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)

        ref_spk_embedding = payload.get("ref_spk_embedding")
        if ref_spk_embedding is None:
            raise ValueError("Saved prompt is missing ref_spk_embedding.")
        if not torch.is_tensor(ref_spk_embedding):
            ref_spk_embedding = torch.tensor(ref_spk_embedding)

        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=bool(payload.get("x_vector_only_mode", False)),
                icl_mode=bool(payload.get("icl_mode", not bool(payload.get("x_vector_only_mode", False)))),
                ref_text=payload.get("ref_text"),
            )
        )
    return items


def _torch_load_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _persist_prompt_to_output_folder(output_folder: str, prompt_items: List[VoiceClonePromptItem], prompt_meta: Dict[str, Any]) -> str:
    paths = _get_batch_paths(output_folder)
    payload = {
        "items": _serialize_prompt_items(prompt_items),
        "meta": prompt_meta,
    }
    torch.save(payload, paths["prompt"])
    return paths["prompt"]


def _load_prompt_from_output_folder(output_folder: str):
    prompt_path = _get_batch_paths(output_folder)["prompt"]
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt checkpoint not found at {prompt_path}")

    payload = _torch_load_compat(prompt_path)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError("Prompt checkpoint is invalid.")

    prompt_items = _deserialize_prompt_items(payload["items"])
    prompt_meta = payload.get("meta", {})
    return prompt_items, prompt_meta


def _summarize_batch(rows: List[Dict[str, Any]], queue_row_indices: List[int]) -> str:
    queue_set = set(queue_row_indices)
    success = 0
    failed = 0
    skipped = 0
    for row in rows:
        if row["row_index"] not in queue_set:
            continue
        status = str(row.get("status", ""))
        if status.startswith("Success"):
            success += 1
        elif status.startswith("Failed"):
            failed += 1
        elif status.startswith("Skipped"):
            skipped += 1
    return f"Summary: {success} success, {failed} failed, {skipped} skipped."


def _serialize_rows_for_state(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized = []
    for row in rows:
        serialized.append(
            {
                "row_index": int(row["row_index"]),
                "script": row.get("script", ""),
                "filename": row.get("filename", ""),
                "status": row.get("status", "Pending"),
                "language": row.get("language"),
                "enabled": bool(row.get("enabled", True)),
                "output_path": row.get("output_path"),
                "error": row.get("error"),
                "chunks": int(row.get("chunks", 0)),
                "note": row.get("note"),
            }
        )
    return serialized


def _write_manifest_row(output_folder: str, row: Dict[str, Any]):
    manifest_path = _get_batch_paths(output_folder)["manifest"]
    file_exists = os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_MANIFEST_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": _utc_now(),
                "row_index": row["row_index"],
                "filename": row.get("filename", ""),
                "status": row.get("status", ""),
                "language": row.get("resolved_language", row.get("language") or ""),
                "chunks": row.get("chunks", 0),
                "output_path": row.get("output_path") or "",
                "message": row.get("note") or row.get("error") or "",
            }
        )


def _write_state_file(
    output_folder: str,
    rows: List[Dict[str, Any]],
    queue_row_indices: List[int],
    next_queue_position: int,
    prompt_meta: Optional[Dict[str, Any]],
    source_file: Optional[str],
    default_language: str,
    model_size: str,
    run_mode: str,
):
    state_path = _get_batch_paths(output_folder)["state"]
    payload = {
        "updated_at": _utc_now(),
        "output_folder": output_folder,
        "source_file": source_file,
        "default_language": default_language,
        "model_size": model_size,
        "run_mode": run_mode,
        "queue_row_indices": queue_row_indices,
        "next_queue_position": next_queue_position,
        "prompt_meta": prompt_meta or {},
        "rows": _serialize_rows_for_state(rows),
        "summary": _summarize_batch(rows, queue_row_indices),
    }
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_state_file(output_folder: str) -> Dict[str, Any]:
    state_path = _get_batch_paths(output_folder)["state"]
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"state.json not found in {output_folder}")
    with open(state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_script_rows(file_path: str):
    if not file_path:
        raise ValueError("Upload a CSV or XLSX file first.")

    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".csv":
        dataframe = pd.read_csv(file_path)
    elif extension == ".xlsx":
        dataframe = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

    if dataframe.empty:
        raise ValueError("The uploaded file is empty.")

    normalized_columns = {str(column).strip().lower(): column for column in dataframe.columns}
    if "script" not in normalized_columns:
        raise ValueError("The uploaded file must contain a 'script' column.")

    working_df = dataframe.copy()
    if "order" in normalized_columns:
        order_column = normalized_columns["order"]
        working_df["__sort_order"] = pd.to_numeric(working_df[order_column], errors="coerce")
        working_df["__original_position"] = np.arange(len(working_df))
        working_df = working_df.sort_values(
            by=["__sort_order", "__original_position"],
            kind="stable",
            na_position="last",
        )

    working_df = working_df.reset_index(drop=True)
    used_names = set()
    total_rows = len(working_df)
    rows = []

    for row_index, (_, dataframe_row) in enumerate(working_df.iterrows(), start=1):
        row = _clone_row_template(row_index)
        row["script"] = _clean_script_text(dataframe_row[normalized_columns["script"]])
        row["language"] = _normalize_language_choice(
            dataframe_row[normalized_columns["language"]],
            fallback=None,
        ) if "language" in normalized_columns else None
        row["enabled"] = _normalize_enabled(
            dataframe_row[normalized_columns["enabled"]]
        ) if "enabled" in normalized_columns else True
        row["filename"] = _make_safe_filename(
            dataframe_row[normalized_columns["filename"]] if "filename" in normalized_columns else None,
            row_index,
            total_rows,
            used_names,
        )

        if not row["enabled"]:
            row["status"] = "Skipped (disabled)"
        elif not row["script"]:
            row["status"] = "Skipped (empty script)"

        rows.append(row)

    return rows


def load_batch_script(file_path):
    try:
        rows = _load_script_rows(file_path)
    except Exception as exc:
        BATCH_SESSION["rows"] = []
        BATCH_SESSION["source_file"] = None
        return [], f"Script load failed: {exc}"

    BATCH_SESSION["rows"] = [dict(row) for row in rows]
    BATCH_SESSION["source_file"] = file_path
    return _rows_to_table_value(rows), f"Loaded {len(rows)} script rows from {os.path.basename(file_path)}."


def build_and_cache_batch_prompt(ref_audio, ref_text, language, model_size, output_folder):
    if not ref_audio:
        return "Prompt status: not cached", "Reference audio is required."

    cleaned_ref_text = _clean_script_text(ref_text)
    if not cleaned_ref_text:
        return "Prompt status: not cached", "Reference text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return "Prompt status: not cached", "Reference audio could not be read."

    fingerprint = _prompt_fingerprint(ref_audio, cleaned_ref_text, model_size)
    prompt_meta = BATCH_SESSION.get("prompt_meta") or {}
    if BATCH_SESSION.get("prompt_items") and prompt_meta.get("fingerprint") == fingerprint:
        prompt_status = _format_prompt_status(prompt_meta, prompt_persisted=False)
        return prompt_status, "Voice prompt already cached in memory."

    try:
        tts = get_model("Base", model_size)
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=cleaned_ref_text,
            x_vector_only_mode=False,
        )
    except Exception as exc:
        return "Prompt status: not cached", f"Failed to build prompt: {exc}"

    prompt_meta = {
        "fingerprint": fingerprint,
        "reference_name": os.path.basename(ref_audio),
        "reference_audio_path": os.path.abspath(ref_audio),
        "reference_text": cleaned_ref_text,
        "model_size": model_size,
        "default_language": _normalize_language_choice(language, fallback="Auto"),
        "built_at": _utc_now(),
    }

    BATCH_SESSION["prompt_items"] = prompt_items
    BATCH_SESSION["prompt_meta"] = prompt_meta

    prompt_persisted = False
    if output_folder and output_folder.strip():
        try:
            normalized_output = _normalize_output_folder_path(output_folder)
            _persist_prompt_to_output_folder(normalized_output, prompt_items, prompt_meta)
            prompt_persisted = True
        except Exception:
            prompt_persisted = False

    return _format_prompt_status(prompt_meta, prompt_persisted), "Voice prompt cached and ready for batch cloning."


def _prepare_prompt_for_new_batch(ref_audio, ref_text, model_size, output_folder: str):
    cleaned_ref_text = _clean_script_text(ref_text)
    expected_fingerprint = _prompt_fingerprint(ref_audio, cleaned_ref_text, model_size)
    prompt_items = BATCH_SESSION.get("prompt_items")
    prompt_meta = BATCH_SESSION.get("prompt_meta") or {}

    if not prompt_items or prompt_meta.get("fingerprint") != expected_fingerprint:
        raise ValueError("Voice prompt cache is missing or stale. Click 'Build & Cache Voice Prompt' first.")

    _persist_prompt_to_output_folder(output_folder, prompt_items, prompt_meta)
    return prompt_items, prompt_meta


def _row_language(row: Dict[str, Any], default_language: str) -> str:
    return _normalize_language_choice(row.get("language"), fallback=_normalize_language_choice(default_language, fallback="Auto"))


def _prepare_batch_queue(rows: List[Dict[str, Any]], selected_row_indices: Optional[List[int]] = None) -> List[int]:
    if selected_row_indices is None:
        return [row["row_index"] for row in rows]
    selected_set = set(selected_row_indices)
    return [row["row_index"] for row in rows if row["row_index"] in selected_set]


def _build_run_rows(table_value, reset_for_new_run: bool = True) -> List[Dict[str, Any]]:
    rows = _sync_batch_rows(table_value)
    if reset_for_new_run:
        rows = _reset_rows_for_new_run(rows)
        BATCH_SESSION["rows"] = [dict(row) for row in rows]
    return rows


def _generate_batch_audio_for_row(
    row: Dict[str, Any],
    prompt_items: List[VoiceClonePromptItem],
    model_size: str,
    default_language: str,
    output_folder: str,
):
    tts = get_model("Base", model_size)
    language = _row_language(row, default_language)
    text_chunks, _ = text_chunk(row["script"], language, char_limit=280)
    chunk_files = []
    final_output_path = os.path.join(output_folder, row["filename"])

    try:
        for chunk_index, chunk in enumerate(text_chunks):
            if BATCH_SESSION["cancel_requested"]:
                raise BatchCancelled()

            wavs, sr = tts.generate_voice_clone(
                text=chunk.strip(),
                language=language,
                voice_clone_prompt=prompt_items,
                max_new_tokens=2048,
            )

            temp_filename = f"temp_batch_clone_{row['row_index']}_{chunk_index}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)

            del wavs
            torch.cuda.empty_cache()
            gc.collect()

        if BATCH_SESSION["cancel_requested"]:
            raise BatchCancelled()

        stitched_path = stitch_chunk_files(chunk_files, final_output_path)
        chunk_files = []
        note = "Auto-split into multiple chunks for stability." if len(text_chunks) > 1 else "Generated from cached voice prompt."
        return stitched_path, len(text_chunks), note, language
    finally:
        for temp_file in chunk_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass


def stop_batch_generation():
    if not BATCH_SESSION["running"]:
        return "No active batch job is running."

    BATCH_SESSION["cancel_requested"] = True
    return "Stop requested. The batch will pause at the next safe checkpoint."


def _run_batch_generator(
    rows: List[Dict[str, Any]],
    queue_row_indices: List[int],
    prompt_items: List[VoiceClonePromptItem],
    prompt_meta: Dict[str, Any],
    output_folder: str,
    default_language: str,
    model_size: str,
    run_mode: str,
    next_queue_position: int = 0,
):
    if not queue_row_indices:
        raise ValueError("There are no rows available to process.")

    BATCH_SESSION["cancel_requested"] = False
    BATCH_SESSION["running"] = True
    BATCH_SESSION["rows"] = [dict(row) for row in rows]
    BATCH_SESSION["prompt_items"] = prompt_items
    BATCH_SESSION["prompt_meta"] = prompt_meta

    rows_by_index = {row["row_index"]: row for row in rows}
    prompt_status = _format_prompt_status(prompt_meta, prompt_persisted=True)
    source_file = BATCH_SESSION.get("source_file")

    try:
        yield _rows_to_table_value(rows), f"Batch ready. Processing {len(queue_row_indices)} rows into {output_folder}", prompt_status

        cursor = next_queue_position
        while cursor < len(queue_row_indices):
            if BATCH_SESSION["cancel_requested"]:
                _write_state_file(
                    output_folder,
                    rows,
                    queue_row_indices,
                    cursor,
                    prompt_meta,
                    source_file,
                    default_language,
                    model_size,
                    run_mode,
                )
                yield _rows_to_table_value(rows), "Batch paused before starting the next row.", prompt_status
                return

            row_index = queue_row_indices[cursor]
            row = rows_by_index[row_index]
            _write_state_file(
                output_folder,
                rows,
                queue_row_indices,
                cursor,
                prompt_meta,
                source_file,
                default_language,
                model_size,
                run_mode,
            )

            if not row.get("enabled", True):
                row["status"] = "Skipped (disabled)"
                row["error"] = None
                row["chunks"] = 0
                row["note"] = "Row disabled by input file."
                _write_manifest_row(output_folder, row)
                cursor += 1
                _write_state_file(
                    output_folder,
                    rows,
                    queue_row_indices,
                    cursor,
                    prompt_meta,
                    source_file,
                    default_language,
                    model_size,
                    run_mode,
                )
                yield _rows_to_table_value(rows), f"Skipped row {row_index}: disabled.", prompt_status
                continue

            if not row.get("script"):
                row["status"] = "Skipped (empty script)"
                row["error"] = None
                row["chunks"] = 0
                row["note"] = "Row has no script text."
                _write_manifest_row(output_folder, row)
                cursor += 1
                _write_state_file(
                    output_folder,
                    rows,
                    queue_row_indices,
                    cursor,
                    prompt_meta,
                    source_file,
                    default_language,
                    model_size,
                    run_mode,
                )
                yield _rows_to_table_value(rows), f"Skipped row {row_index}: empty script.", prompt_status
                continue

            row["status"] = "Running"
            yield _rows_to_table_value(rows), f"Generating row {cursor + 1}/{len(queue_row_indices)} -> {row['filename']}", prompt_status

            try:
                output_path, chunk_count, note, resolved_language = _generate_batch_audio_for_row(
                    row=row,
                    prompt_items=prompt_items,
                    model_size=model_size,
                    default_language=default_language,
                    output_folder=output_folder,
                )
                row["status"] = "Success"
                row["resolved_language"] = resolved_language
                row["output_path"] = output_path
                row["chunks"] = chunk_count
                row["error"] = None
                row["note"] = note
                cursor += 1
                _write_manifest_row(output_folder, row)
                _write_state_file(
                    output_folder,
                    rows,
                    queue_row_indices,
                    cursor,
                    prompt_meta,
                    source_file,
                    default_language,
                    model_size,
                    run_mode,
                )
                yield _rows_to_table_value(rows), f"Saved {row['filename']} ({cursor}/{len(queue_row_indices)}).", prompt_status
            except BatchCancelled:
                row["status"] = "Pending"
                row["note"] = "Cancelled before row completion."
                _write_state_file(
                    output_folder,
                    rows,
                    queue_row_indices,
                    cursor,
                    prompt_meta,
                    source_file,
                    default_language,
                    model_size,
                    run_mode,
                )
                yield _rows_to_table_value(rows), f"Batch paused. Resume will restart from row {row_index}.", prompt_status
                return
            except Exception as exc:
                row["status"] = "Failed"
                row["output_path"] = None
                row["chunks"] = 0
                row["error"] = str(exc)
                row["note"] = "Generation failed."
                cursor += 1
                _write_manifest_row(output_folder, row)
                _write_state_file(
                    output_folder,
                    rows,
                    queue_row_indices,
                    cursor,
                    prompt_meta,
                    source_file,
                    default_language,
                    model_size,
                    run_mode,
                )
                yield _rows_to_table_value(rows), f"Row {row_index} failed: {exc}", prompt_status

        yield _rows_to_table_value(rows), f"Batch finished. {_summarize_batch(rows, queue_row_indices)}", prompt_status
    finally:
        BATCH_SESSION["running"] = False
        BATCH_SESSION["cancel_requested"] = False
        BATCH_SESSION["rows"] = [dict(row) for row in rows]


def generate_all_batch_rows(table_value, output_folder, default_language, model_size, ref_audio, ref_text):
    try:
        normalized_output = _normalize_output_folder_path(output_folder)
        rows = _build_run_rows(table_value, reset_for_new_run=True)
        prompt_items, prompt_meta = _prepare_prompt_for_new_batch(ref_audio, ref_text, model_size, normalized_output)
        queue_row_indices = _prepare_batch_queue(rows)

        manifest_path = _get_batch_paths(normalized_output)["manifest"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=BATCH_MANIFEST_HEADERS)
            writer.writeheader()

        _write_state_file(
            normalized_output,
            rows,
            queue_row_indices,
            0,
            prompt_meta,
            BATCH_SESSION.get("source_file"),
            default_language,
            model_size,
            "all",
        )
    except Exception as exc:
        yield _rows_to_table_value(BATCH_SESSION["rows"]), f"Batch start failed: {exc}", _format_prompt_status(BATCH_SESSION.get("prompt_meta"))
        return

    yield from _run_batch_generator(
        rows=rows,
        queue_row_indices=queue_row_indices,
        prompt_items=prompt_items,
        prompt_meta=prompt_meta,
        output_folder=normalized_output,
        default_language=default_language,
        model_size=model_size,
        run_mode="all",
        next_queue_position=0,
    )


def generate_selected_batch_rows(table_value, output_folder, default_language, model_size, ref_audio, ref_text, selection_text):
    try:
        normalized_output = _normalize_output_folder_path(output_folder)
        rows = _build_run_rows(table_value, reset_for_new_run=True)
        selected_indices = _parse_selection_text(selection_text, [row["row_index"] for row in rows])
        prompt_items, prompt_meta = _prepare_prompt_for_new_batch(ref_audio, ref_text, model_size, normalized_output)
        queue_row_indices = _prepare_batch_queue(rows, selected_indices)

        manifest_path = _get_batch_paths(normalized_output)["manifest"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=BATCH_MANIFEST_HEADERS)
            writer.writeheader()

        _write_state_file(
            normalized_output,
            rows,
            queue_row_indices,
            0,
            prompt_meta,
            BATCH_SESSION.get("source_file"),
            default_language,
            model_size,
            "selected",
        )
    except Exception as exc:
        yield _rows_to_table_value(BATCH_SESSION["rows"]), f"Batch start failed: {exc}", _format_prompt_status(BATCH_SESSION.get("prompt_meta"))
        return

    yield from _run_batch_generator(
        rows=rows,
        queue_row_indices=queue_row_indices,
        prompt_items=prompt_items,
        prompt_meta=prompt_meta,
        output_folder=normalized_output,
        default_language=default_language,
        model_size=model_size,
        run_mode="selected",
        next_queue_position=0,
    )


def resume_batch_from_checkpoint(output_folder):
    try:
        normalized_output = _normalize_output_folder_path(output_folder)
        state = _load_state_file(normalized_output)
        rows = state.get("rows", [])
        if not rows:
            raise ValueError("Checkpoint has no rows to resume.")

        prompt_items, prompt_meta = _load_prompt_from_output_folder(normalized_output)
        queue_row_indices = [int(item) for item in state.get("queue_row_indices", [])]
        next_queue_position = int(state.get("next_queue_position", 0))
        default_language = state.get("default_language", "Auto")
        model_size = state.get("model_size", "1.7B")

        BATCH_SESSION["source_file"] = state.get("source_file")
        BATCH_SESSION["rows"] = [dict(row) for row in rows]
    except Exception as exc:
        yield _rows_to_table_value(BATCH_SESSION["rows"]), f"Resume failed: {exc}", _format_prompt_status(BATCH_SESSION.get("prompt_meta"))
        return

    yield from _run_batch_generator(
        rows=[dict(row) for row in rows],
        queue_row_indices=queue_row_indices,
        prompt_items=prompt_items,
        prompt_meta=prompt_meta,
        output_folder=normalized_output,
        default_language=default_language,
        model_size=model_size,
        run_mode=state.get("run_mode", "resume"),
        next_queue_position=next_queue_position,
    )


# --- UI Construction ---

def on_mode_change(mode):
    return gr.update(visible=("High-Quality" in mode))

def build_ui():
    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;} .tab-content {padding: 20px;}"

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo") as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ Qwen3-TTS </h1>
            <a href="https://colab.research.google.com/github/NeuralFalconYT/Qwen3-TTS-Colab/blob/main/Qwen3_TTS_Colab.ipynb" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4285F4; color: white; border-radius: 6px; text-decoration: none; font-size: 1em;">🥳 Run on Google Colab</a>
        </div>""")

        with gr.Tabs():
            # --- Tab 1: Voice Design ---
            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(label="Text to Synthesize", lines=4, value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                                                 placeholder="Enter the text you want to convert to speech...")
                        design_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                        design_instruct = gr.Textbox(label="Voice Description", lines=3,  placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.")
                        design_btn = gr.Button("Generate with Custom Voice", variant="primary")
                        with gr.Accordion("More options", open=False):
                            with gr.Row():
                              design_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                              design_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                        
                        

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        design_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                d_srt1 = gr.File(label="Original (Whisper)")
                                d_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                d_srt3 = gr.File(label="Word-level")
                                d_srt4 = gr.File(label="Shorts/Reels")

                design_btn.click(
                    generate_voice_design, 
                    inputs=[design_text, design_language, design_instruct, design_rem_silence, design_make_subs], 
                    outputs=[design_audio_out, design_status, d_srt1, d_srt2, d_srt3, d_srt4]
                )

            # --- Tab 2: Voice Clone ---
            with gr.Tab("Voice Clone (Base)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(label="Target Text", lines=3, placeholder="Enter the text you want the cloned voice to speak...")
                        clone_ref_audio = gr.Audio(label="Reference Audio (Upload a voice sample to clone)", type="filepath")
                        
                        with gr.Row():
                            clone_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto",scale=1)
                            clone_model_size = gr.Dropdown(label="Model Size", choices=MODEL_SIZES, value="1.7B",scale=1)
                            clone_mode = gr.Dropdown(
                                label="Mode",
                                choices=["High-Quality (Audio + Transcript)", "Fast (Audio Only)"],
                                value="High-Quality (Audio + Transcript)",
                                interactive=True,
                                scale=2
                            )
                        
                        clone_ref_text = gr.Textbox(label="Reference Text", lines=2, visible=True)
                        clone_btn = gr.Button("Clone & Generate", variant="primary")
                        with gr.Accordion("More options", open=False):
                            with gr.Row():
                              clone_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                              clone_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)

                        

                    with gr.Column(scale=2):
                        clone_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        clone_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                c_srt1 = gr.File(label="Original")
                                c_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                c_srt3 = gr.File(label="Word-level")
                                c_srt4 = gr.File(label="Shorts/Reels")

                clone_mode.change(on_mode_change, inputs=[clone_mode], outputs=[clone_ref_text])
                clone_ref_audio.change(transcribe_reference, inputs=[clone_ref_audio, clone_mode, clone_language], outputs=[clone_ref_text])
                
                clone_btn.click(
                    smart_generate_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_mode, clone_model_size, clone_rem_silence, clone_make_subs],
                    outputs=[clone_audio_out, clone_status, c_srt1, c_srt2, c_srt3, c_srt4]
                )

            # --- Tab 3: TTS (CustomVoice) ---
            with gr.Tab("TTS (CustomVoice)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(label="Text", lines=4,   placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities.")
                        with gr.Row():
                            tts_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="English")
                            tts_speaker = gr.Dropdown(label="Speaker", choices=SPEAKERS, value="Ryan")
                        with gr.Row():
                            tts_instruct = gr.Textbox(label="Style Instruction (Optional)", lines=2,placeholder="e.g., Speak in a cheerful and energetic tone")
                            tts_model_size = gr.Dropdown(label="Size", choices=MODEL_SIZES, value="1.7B")
                        tts_btn = gr.Button("Generate Speech", variant="primary")
                        with gr.Accordion("More options", open=False):
                            with gr.Row():
                              tts_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                              tts_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                            
                        

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        tts_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                t_srt1 = gr.File(label="Original")
                                t_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                t_srt3 = gr.File(label="Word-level")
                                t_srt4 = gr.File(label="Shorts/Reels")

                tts_btn.click(
                    generate_custom_voice, 
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_rem_silence, tts_make_subs], 
                    outputs=[tts_audio_out, tts_status, t_srt1, t_srt2, t_srt3, t_srt4]
                )
            # --- Tab 4: Batch Voice Clone Studio ---
            with gr.Tab("Batch Voice Clone Studio"):
                gr.Markdown("""
                Upload a CSV/XLSX script, cache a single base-model voice prompt, then generate one cloned `.wav` per row with checkpoint-safe resume support.
                """)

                with gr.Group():
                    gr.Markdown("### Voice Prompt Setup")
                    with gr.Row():
                        batch_ref_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                        )
                        with gr.Column():
                            batch_ref_text = gr.Textbox(
                                label="Reference Text",
                                lines=3,
                                placeholder="Transcript of the reference audio used for ICL voice cloning.",
                            )
                            with gr.Row():
                                batch_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                                batch_model_size = gr.Dropdown(label="Model Size", choices=MODEL_SIZES, value="1.7B")
                            batch_build_prompt_btn = gr.Button("Build & Cache Voice Prompt", variant="primary")
                            batch_prompt_status = gr.Textbox(
                                label="Prompt Cache Status",
                                value="Prompt status: not cached",
                                interactive=False,
                            )

                with gr.Group():
                    gr.Markdown("### Script Upload & Edit")
                    batch_script_upload = gr.File(
                        label="Upload Script (.csv or .xlsx)",
                        file_types=[".csv", ".xlsx"],
                        type="filepath",
                    )
                    batch_table = gr.Dataframe(
                        headers=BATCH_TABLE_HEADERS,
                        datatype=["number", "str", "str", "str"],
                        value=[],
                        row_count=(0, "dynamic"),
                        col_count=(len(BATCH_TABLE_HEADERS), "fixed"),
                        interactive=True,
                        type="array",
                        label="Editable Script Table",
                    )

                with gr.Group():
                    gr.Markdown("### Batch Controls & Output Setup")
                    batch_output_folder = gr.Textbox(
                        label="Output Folder Path",
                        placeholder="/content/drive/MyDrive/MyAudioFolder",
                    )
                    batch_selected_rows = gr.Textbox(
                        label="Selected Row Indexes (for Generate Selected)",
                        placeholder="Example: 1, 3-5, 8",
                    )
                    with gr.Row():
                        batch_generate_all_btn = gr.Button("Generate All", variant="primary")
                        batch_generate_selected_btn = gr.Button("Generate Selected")
                        batch_stop_btn = gr.Button("Stop/Cancel")
                        batch_resume_btn = gr.Button("Resume from Last Checkpoint")
                    batch_status = gr.Textbox(
                        label="Batch Status",
                        interactive=False,
                        lines=4,
                        value="Load a script and cache a voice prompt to begin.",
                    )

                batch_script_upload.change(
                    load_batch_script,
                    inputs=[batch_script_upload],
                    outputs=[batch_table, batch_status],
                    queue=False,
                )

                batch_build_prompt_btn.click(
                    build_and_cache_batch_prompt,
                    inputs=[batch_ref_audio, batch_ref_text, batch_language, batch_model_size, batch_output_folder],
                    outputs=[batch_prompt_status, batch_status],
                    queue=False,
                )

                batch_generate_all_btn.click(
                    generate_all_batch_rows,
                    inputs=[batch_table, batch_output_folder, batch_language, batch_model_size, batch_ref_audio, batch_ref_text],
                    outputs=[batch_table, batch_status, batch_prompt_status],
                )

                batch_generate_selected_btn.click(
                    generate_selected_batch_rows,
                    inputs=[batch_table, batch_output_folder, batch_language, batch_model_size, batch_ref_audio, batch_ref_text, batch_selected_rows],
                    outputs=[batch_table, batch_status, batch_prompt_status],
                )

                batch_stop_btn.click(
                    stop_batch_generation,
                    outputs=[batch_status],
                    queue=False,
                )

                batch_resume_btn.click(
                    resume_batch_from_checkpoint,
                    inputs=[batch_output_folder],
                    outputs=[batch_table, batch_status, batch_prompt_status],
                )

            # --- Tab 5: About ---
            with gr.Tab("About"):
                gr.Markdown("""
                # Qwen3-TTS 
                A unified Text-to-Speech demo featuring three powerful modes:
                - **Voice Design**: Create custom voices using natural language descriptions
                - **Voice Clone (Base)**: Clone any voice from a reference audio
                - **TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions

                Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
                """)

                gr.HTML("""
                <hr>
                <p style="color: red; font-weight: bold; font-size: 16px;">
                ⚠️ NOTE
                </p>
                <p>
                This Gradio UI is not affiliated with the official Qwen3-TTS project and is based on the
                official Qwen3-TTS demo UI:<br>
                <a href="https://huggingface.co/spaces/Qwen/Qwen3-TTS" target="_blank">
                https://huggingface.co/spaces/Qwen/Qwen3-TTS
                </a>
                </p>

                <p><b>Additional features:</b></p>
                <ul>
                  <li>Automatic transcription support using faster-whisper-large-v3-turbo-ct2</li>
                  <li>Long text input support</li>
                  <li>Because we are using Whisper, subtitles are also added</li>
                </ul>
                """)

             
    return demo

# if __name__ == "__main__":
#     demo = build_ui()
#     demo.launch(share=True, debug=True)



import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(share,debug):
    demo = build_ui()
    # demo.launch(share=True, debug=True)
    demo.queue().launch(share=share,debug=debug)

if __name__ == "__main__":
    main()    
