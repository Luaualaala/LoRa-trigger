"""LoRA Trigger Tap - trigger-word companion for ComfyUI LoRA loaders.

Maxed-Out-99/ComfyUI-MaxedOut's "Lora Loader MXD" is the primary integration,
with a loader-agnostic fallback for built-in, stacked, and other power loaders.

Passes `model`/`clip` straight through and adds a `triggers` STRING output: the
trigger words for whichever LoRAs are currently enabled upstream, resolved from
local metadata (no CivitAI call unless you click a per-LoRA "Fetch" button).

This node deliberately does NOT import the LoRA loader's code directly - that would
only work if this plugin were nested inside the loader's own repo. Instead it looks
up the loader's registered class through ComfyUI's shared `nodes.NODE_CLASS_MAPPINGS`
at *execution time* (after all custom nodes have finished loading), which is the
standard, decoupled way one ComfyUI plugin can call into another's node class. If
the loader isn't installed, this node just logs a warning and outputs no triggers
rather than failing to import.

Trigger-word resolution, tiers 1-3 (local reads only, no network):
  1. `.cm-info.json` sidecar next to the .safetensors (Stability Matrix's Checkpoint
     Manager convention) - the model author's own curated CivitAI trigger words.
  2. `modelspec.trigger_phrase` in the safetensors header (SAI ModelSpec 1.0.1,
     embedded by kohya-ss/sd-scripts when the trainer sets it).
  3. `caption_prefix` from `ss_datasets[*].subsets[*].caption_prefix` in the same
     header - the literal string the training config prepended to every caption.
Tier 4 (manual, per-LoRA "Fetch" button, CivitAI's hash-lookup API) is handled by the
aiohttp routes below and caches into the same `.cm-info.json` tier 1 reads from.
"""

import asyncio
import hashlib
import json
import os
import time

import aiohttp
import folder_paths
import nodes
from aiohttp import web
from server import PromptServer

MXD_LORA_LOADER_CLASS_TYPE = "Lora Loader MXD"
NODE_NAME = "LoRA Trigger Tap"
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
_LORA_NAME_KEYS = ("lora", "lora_name", "filename", "file", "name")
_ENABLED_KEYS = ("on", "enabled", "active")
_STRENGTH_KEYS = (
  "strength", "strength_model", "strength_clip", "model_strength", "clip_strength",
  "weight",
)


class _AnyType(str):
  """Always equal in inequality comparisons, so ComfyUI's input-type checks never
  reject it. Common community pattern (credit: pythongosssss), not MXD-specific.
  """

  def __ne__(self, __value: object) -> bool:
    return False


class _FlexibleOptionalInputType(dict):
  """Lets INPUT_TYPES declare a fixed set of optional inputs (model/clip) while still
  accepting an arbitrary number of dynamically-named ones (trigger_1, trigger_2, ...)
  without ComfyUI's validation rejecting the extras. Same community pattern as above.
  """

  def __init__(self, any_type, data=None):
    super().__init__()
    self.any_type = any_type
    self.data = data
    if self.data is not None:
      for key, value in self.data.items():
        self[key] = value

  def __getitem__(self, key):
    if self.data is not None and key in self.data:
      return self.data[key]
    return (self.any_type,)

  def __contains__(self, key):
    return True


_any_type = _AnyType("*")


def _log(message: str):
  print(f"[LoRA Trigger Tap] {message}")


# ---------------------------------------------------------------------------
# Small local JSON/file helpers (kept self-contained rather than depending on
# another plugin's utils module, since this node is meant to install standalone).
# ---------------------------------------------------------------------------

def _load_json_file(file_path, default=None):
  if not file_path or not os.path.isfile(file_path):
    return default
  with open(file_path, "r", encoding="UTF-8") as file:
    try:
      return json.loads(file.read())
    except json.JSONDecodeError:
      return default


def _save_json_file(file_path, data):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, "w+", encoding="UTF-8") as file:
    json.dump(data, file, sort_keys=False, indent=2, separators=(",", ": "))


def get_lora_file_path(lora_filename):
  """Resolves a LoRA's on-disk path, or None if it can't be found."""
  path = folder_paths.get_full_path("loras", lora_filename)
  return path if path and os.path.isfile(path) else None


def cm_info_sidecar_path(file_path):
  return f'{os.path.splitext(file_path)[0]}.cm-info.json'


# ---------------------------------------------------------------------------
# Tier 1-3 trigger-word resolution (local only, no network).
# ---------------------------------------------------------------------------

def read_cm_info_sidecar(file_path):
  """Tier 1: reads the `.cm-info.json` sidecar, if present."""
  data = _load_json_file(cm_info_sidecar_path(file_path), default=None)
  if not data:
    return None
  raw_words = data.get("TrainedWords") or data.get("trainedWords") or []
  return _normalize_trigger_words(raw_words) or None


def _normalize_trigger_words(raw_words):
  """Normalizes string/list metadata without splitting one string into characters."""
  if isinstance(raw_words, str):
    raw_words = [raw_words]
  if not isinstance(raw_words, (list, tuple)):
    return []
  words = []
  for entry in raw_words:
    if isinstance(entry, str) and entry.strip():
      words.append(entry.strip())
    elif isinstance(entry, dict) and entry.get("word"):
      words.append(str(entry["word"]).strip())
  return words


def read_safetensors_header_metadata(file_path):
  """Reads the `__metadata__` block from a safetensors file's own header."""
  if not file_path.endswith(".safetensors"):
    return {}
  try:
    with open(file_path, "rb") as file:
      header_size = int.from_bytes(file.read(8), "little", signed=False)
      if header_size <= 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
        return {}
      header_json = json.loads(file.read(header_size))
  except (OSError, ValueError, json.JSONDecodeError):
    return {}
  return header_json.get("__metadata__") or {}


def _trigger_phrase_from_metadata(metadata):
  """Tier 2: `modelspec.trigger_phrase`."""
  phrase = metadata.get("modelspec.trigger_phrase")
  if isinstance(phrase, str) and phrase.strip():
    return [phrase.strip()]
  return None


def _caption_prefixes_from_metadata(metadata):
  """Tier 3: `ss_datasets[*].subsets[*].caption_prefix`."""
  raw = metadata.get("ss_datasets")
  if not raw:
    return None
  datasets = raw
  if isinstance(raw, str):
    try:
      datasets = json.loads(raw)
    except json.JSONDecodeError:
      return None
  if not isinstance(datasets, list):
    return None

  prefixes = []
  for dataset in datasets:
    subsets = dataset.get("subsets", []) if isinstance(dataset, dict) else []
    for subset in subsets:
      prefix = subset.get("caption_prefix") if isinstance(subset, dict) else None
      if isinstance(prefix, str):
        cleaned = prefix.strip().rstrip(",").strip()
        if cleaned and cleaned not in prefixes:
          prefixes.append(cleaned)
  return prefixes or None


def get_trigger_word_tiers(lora_filename, max_words=None):
  """Resolves trigger words for a LoRA via tiers 1-3, in order.

  Returns (tier, words) where tier is 1/2/3 for whichever tier first produced a
  non-empty result, or (None, []) if nothing local was found (tier 4/5 territory).
  """
  file_path = get_lora_file_path(lora_filename)
  if file_path is None:
    return (None, [])

  cm_info_words = read_cm_info_sidecar(file_path)
  if cm_info_words:
    return (1, cm_info_words[:max_words] if max_words else cm_info_words)

  metadata = read_safetensors_header_metadata(file_path)

  trigger_phrase = _trigger_phrase_from_metadata(metadata)
  if trigger_phrase:
    return (2, trigger_phrase[:max_words] if max_words else trigger_phrase)

  caption_prefixes = _caption_prefixes_from_metadata(metadata)
  if caption_prefixes:
    return (3, caption_prefixes[:max_words] if max_words else caption_prefixes)

  return (None, [])


def resolve_local_trigger_words(lora_filename, max_words=1):
  """Convenience wrapper used at graph-execution time: just the words, tier-agnostic."""
  _, words = get_trigger_word_tiers(lora_filename, max_words=max_words)
  return words


# ---------------------------------------------------------------------------
# Reading connected LoRA loaders' live enabled/disabled state.
#
# MXD is deliberately first: we call its registered class's own extractor. Other
# loaders are normalized from common prompt formats without importing their plugin.
# ---------------------------------------------------------------------------

def _clean_lora_name(value):
  if not isinstance(value, str):
    return None
  value = value.strip()
  if not value or value.lower() in {"none", "null", "undefined"}:
    return None
  return value


def _entry_is_enabled(entry):
  for key in _ENABLED_KEYS:
    if key in entry and not bool(entry[key]):
      return False
  if entry.get("bypass") is True or entry.get("muted") is True:
    return False
  strengths = [entry[key] for key in _STRENGTH_KEYS if key in entry]
  numeric = [value for value in strengths if isinstance(value, (int, float))]
  return not numeric or any(value != 0 for value in numeric)


def _normalize_lora_entry(value):
  """Returns a common {name, strength?} record for a loader-specific value."""
  if isinstance(value, str):
    name = _clean_lora_name(value)
    return {"name": name} if name else None
  if isinstance(value, (list, tuple)) and value:
    # Common stack convention: [filename, model_strength, clip_strength].
    name = _clean_lora_name(value[0])
    if name:
      strengths = [v for v in value[1:3] if isinstance(v, (int, float))]
      if strengths and not any(v != 0 for v in strengths):
        return None
      return {"name": name, "strength": strengths[0] if strengths else None}
    return None
  if not isinstance(value, dict) or not _entry_is_enabled(value):
    return None
  name = next((_clean_lora_name(value.get(key)) for key in _LORA_NAME_KEYS
               if _clean_lora_name(value.get(key))), None)
  if not name:
    return None
  strength = next((value[key] for key in _STRENGTH_KEYS
                   if isinstance(value.get(key), (int, float))), None)
  result = {"name": name}
  if strength is not None:
    result["strength"] = strength
  return result


def _is_prompt_link(value, prompt=None):
  """Distinguishes Comfy graph links from [filename, strength] stack tuples."""
  if not (isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)):
    return False
  upstream_id = value[0]
  if prompt is not None and str(upstream_id) in prompt:
    return True
  return isinstance(upstream_id, int) or (
    isinstance(upstream_id, str) and upstream_id.isdigit()
  )


def _generic_enabled_loras(prompt_node, prompt=None):
  """Understands built-in, dynamic-widget, numbered-field, and stack formats."""
  inputs = prompt_node.get("inputs", {})
  enabled = []

  # Built-in Load LoRA and compatible single-loader nodes.
  direct_name = _clean_lora_name(inputs.get("lora_name"))
  if direct_name:
    direct = {
      "lora_name": direct_name,
      "strength_model": inputs.get("strength_model"),
      "strength_clip": inputs.get("strength_clip"),
    }
    normalized = _normalize_lora_entry(direct)
    if normalized:
      enabled.append(normalized)

  for input_name, value in inputs.items():
    key = input_name.lower()
    if input_name == "lora_name":
      continue
    if _is_prompt_link(value, prompt):
      continue
    if isinstance(value, dict):
      normalized = _normalize_lora_entry(value)
      if normalized and ("lora" in key or any(k in value for k in _LORA_NAME_KEYS)):
        enabled.append(normalized)
    elif isinstance(value, (list, tuple)) and "lora" in key:
      # Either one tuple entry or a list of stack entries.
      candidates = value if value and isinstance(value[0], (dict, list, tuple)) else [value]
      for candidate in candidates:
        normalized = _normalize_lora_entry(candidate)
        if normalized:
          enabled.append(normalized)
    elif isinstance(value, str) and "lora" in key and (
        "name" in key or "file" in key or key.startswith("lora_")):
      normalized = _normalize_lora_entry(value)
      if normalized:
        # Numbered flat loaders often keep the matching strength in a sibling key.
        suffix = "".join(ch for ch in key if ch.isdigit())
        strength = next((
          inputs.get(strength_key) for strength_key in (
            f"strength_{suffix}", f"strength_model_{suffix}", f"lora_strength_{suffix}"
          ) if isinstance(inputs.get(strength_key), (int, float))
        ), None)
        if strength == 0:
          continue
        if strength is not None:
          normalized["strength"] = strength
        enabled.append(normalized)

  # Avoid double-counting the same field while retaining genuinely repeated LoRAs
  # from separate loader nodes.
  unique = []
  seen = set()
  for entry in enabled:
    if entry["name"] not in seen:
      seen.add(entry["name"])
      unique.append(entry)
  return unique


def get_enabled_loras(prompt_node, prompt=None):
  class_type = prompt_node.get("class_type")
  loader_class = nodes.NODE_CLASS_MAPPINGS.get(class_type)
  get_enabled = getattr(loader_class, "get_enabled_loras_from_prompt_node", None)

  # First-class MXD path gets first refusal; any published compatible extractor
  # follows the same guarded path. A plugin version mismatch must not abort a graph.
  if callable(get_enabled):
    try:
      extracted = get_enabled(prompt_node)
      if extracted is not None:
        return extracted
    except Exception as exc:  # Third-party extractors can fail in plugin-specific ways.
      _log(f"{class_type}'s LoRA extractor failed ({exc}); using generic compatibility.")
  return _generic_enabled_loras(prompt_node, prompt=prompt)


def _is_loader_path_input(input_name):
  """MODEL/CLIP chains plus separate LORA_STACK-style provider links."""
  name = input_name.lower()
  return any(token in name for token in ("model", "clip", "lora"))


class MxdLoraTriggerTap:
  """See module docstring for the full design."""

  NAME = NODE_NAME
  CATEGORY = "mxd"

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable=invalid-name,missing-function-docstring
    return {
      "required": {},
      "optional": _FlexibleOptionalInputType(_any_type, data={
        "model": ("MODEL",),
        "clip": ("CLIP",),
      }),
      "hidden": {
        "prompt": "PROMPT",
        "unique_id": "UNIQUE_ID",
      },
    }

  RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
  RETURN_NAMES = ("MODEL", "CLIP", "triggers", "lora_info")
  FUNCTION = "get_triggers"

  def get_triggers(self, model=None, clip=None, prompt=None, unique_id=None, **kwargs):
    trigger_entries = {}
    for key, value in kwargs.items():
      if not key.upper().startswith("TRIGGER_"):
        continue
      if not isinstance(value, dict) or not value.get("lora"):
        continue
      trigger_entries[value["lora"]] = value

    enabled_loras = []
    loader_nodes = self._find_connected_loader_nodes(prompt, unique_id)
    if loader_nodes:
      for loader_node in loader_nodes:
        enabled_loras.extend(get_enabled_loras(loader_node, prompt=prompt))
    elif prompt is not None:
      _log("No compatible LoRA loader was found upstream; no triggers resolved.")

    # Live-enabled state always comes from the loader's own prompt-node dict
    # (staleness-proof: a LoRA toggled on there is picked up here even if Refresh was
    # never clicked in this node). Word *selection* prefers this node's own persisted
    # checkboxes (matched by LoRA filename, not position) when present, since that's
    # the only way to honor a non-default pick among multiple trigger words.
    #
    # `category` (none/character/style/pose/clothes) is set per-LoRA from this node's
    # own UI (the small badge next to each LoRA row) and passed through unchanged into
    # `lora_info` so a downstream consumer (e.g. Prompt Forge) can route each LoRA's
    # trigger words into the matching field instead of dumping everything into one
    # undifferentiated string.
    words_out = []
    seen = set()
    lora_info = []
    for lora_entry in enabled_loras:
      lora_name = lora_entry.get("name") or lora_entry.get("lora")
      if not lora_name:
        continue
      stored = trigger_entries.get(lora_name)
      picked = []
      category = "none"
      if stored is not None:
        picked = [
          w.get("word") for w in stored.get("words", [])
          if w.get("checked") and w.get("word")
        ]
        category = stored.get("category") or "none"
      if not picked:
        picked = resolve_local_trigger_words(lora_name, max_words=1)
      lora_info.append({"name": lora_name, "category": category, "triggers": picked})
      for word in picked:
        if word not in seen:
          seen.add(word)
          words_out.append(word)

    triggers = ", ".join(words_out)
    return (model, clip, triggers, json.dumps(lora_info))

  @staticmethod
  def _find_connected_loader_nodes(prompt, unique_id):
    """Finds compatible LoRA loaders upstream, including chained built-in nodes.

    Traversal follows MODEL and CLIP prompt links, visits dependencies first to
    preserve application order, and accepts pass-through nodes between loader/tap.
    """
    if not prompt or unique_id is None:
      return []
    this_node = prompt.get(str(unique_id))
    if this_node is None:
      return []

    found = []
    visited = set()

    def visit(node_id):
      node_id = str(node_id)
      if node_id in visited:
        return
      visited.add(node_id)
      node = prompt.get(node_id)
      if not isinstance(node, dict):
        return
      for input_name, value in node.get("inputs", {}).items():
        if not _is_loader_path_input(input_name):
          continue
        if isinstance(value, list) and len(value) == 2:
          visit(value[0])
      if get_enabled_loras(node, prompt=prompt):
        found.append(node)

    for input_name, link in this_node.get("inputs", {}).items():
      if _is_loader_path_input(input_name) and isinstance(link, list) and len(link) == 2:
        visit(link[0])
    return found


NODE_CLASS_MAPPINGS = {
  MxdLoraTriggerTap.NAME: MxdLoraTriggerTap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  MxdLoraTriggerTap.NAME: "LoRA Trigger Tap",
}


# ---------------------------------------------------------------------------
# aiohttp routes: Refresh (tiers 1-3, local reads) and the per-LoRA CivitAI fetch
# (tier 4, manual, async, sequential, 429-aware).
# ---------------------------------------------------------------------------

routes = PromptServer.instance.routes

CIVITAI_HASH_URL = "https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"

# CivitAI-fronting DDoS protection treats request bursts from one IP as an attack
# signature; a single shared lock keeps fetches sequential even if the UI ever grows a
# "fetch all missing" bulk action on top of the current one-button-per-LoRA design.
_civitai_fetch_lock = asyncio.Lock()


def _get_param(request, param, default=None):
  return request.rel_url.query.get(param, default)


def _get_params(request, param):
  getall = getattr(request.rel_url.query, "getall", None)
  return list(getall(param, [])) if callable(getall) else []


def _sha256_of_file(file_path):
  sha256 = hashlib.sha256()
  with open(file_path, "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 128), b""):
      sha256.update(chunk)
  return sha256.hexdigest()


@routes.get("/loratriggertap/api/resolve")
async def api_resolve_triggers(request):
  """Refresh: tiers 1-3 only, local reads, no network call."""
  files = [f for f in _get_params(request, "file") if f]
  if not files:
    # Backward compatibility with clients from before repeated parameters.
    files_param = _get_param(request, "files")
    files = [f for f in (files_param.split(",") if files_param else []) if f]
  data = {}
  for file in files:
    tier, words = await asyncio.to_thread(get_trigger_word_tiers, file)
    data[file] = {"tier": tier, "words": words}
  return web.json_response({"status": 200, "data": data})


@routes.get("/loratriggertap/api/fetch_civitai")
async def api_fetch_civitai(request):
  """Tier 4: manual, per-LoRA CivitAI fetch. Caches into the same `.cm-info.json`
  sidecar tier 1 reads from.
  """
  file_param = _get_param(request, "file")
  if not file_param:
    return web.json_response({"status": 400, "error": "No file provided"})

  file_path = get_lora_file_path(file_param)
  if file_path is None:
    return web.json_response({"status": 404, "error": f"LoRA not found: {file_param}"})

  async with _civitai_fetch_lock:
    try:
      file_hash = await asyncio.to_thread(_sha256_of_file, file_path)
    except OSError as exc:
      return web.json_response({"status": 500, "error": f"Could not hash LoRA: {exc}"})
    api_url = CIVITAI_HASH_URL.format(file_hash=file_hash)
    try:
      # aiohttp's async client, not `requests` - a blocking call here would stall
      # ComfyUI's whole server for the request's duration, not just this node.
      async with aiohttp.ClientSession() as session:
        async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
          if response.status == 429:
            return web.json_response({
              "status": 429,
              "error": "Rate limited by CivitAI",
              "retryAfter": response.headers.get("Retry-After"),
            })
          if response.status != 200:
            return web.json_response({
              "status": response.status,
              "error": f"CivitAI returned HTTP {response.status}",
            })
          data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
      return web.json_response({"status": 502, "error": str(exc)})

  trained_words = _normalize_trigger_words(data.get("trainedWords") or [])

  sidecar_path = cm_info_sidecar_path(file_path)
  cm_info = _load_json_file(sidecar_path, default={}) or {}
  cm_info["TrainedWords"] = trained_words
  cm_info["civitaiModelId"] = data.get("modelId")
  cm_info["civitaiModelVersionId"] = data.get("id")
  cm_info["fetchedAt"] = time.time()
  try:
    await asyncio.to_thread(_save_json_file, sidecar_path, cm_info)
  except OSError as exc:
    return web.json_response({"status": 500, "error": f"Could not save trigger metadata: {exc}"})

  return web.json_response({"status": 200, "data": {"words": trained_words}})
