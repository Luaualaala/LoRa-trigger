"""LoRA Trigger Tap - a companion node for rgthree-style "Power LoRA Loader" nodes
(tested against Maxed-Out-99/ComfyUI-MaxedOut's "Lora Loader MXD").

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
  words = []
  for entry in raw_words:
    if isinstance(entry, str) and entry.strip():
      words.append(entry.strip())
    elif isinstance(entry, dict) and entry.get("word"):
      words.append(str(entry["word"]).strip())
  return words or None


def read_safetensors_header_metadata(file_path):
  """Reads the `__metadata__` block from a safetensors file's own header."""
  if not file_path.endswith(".safetensors"):
    return {}
  try:
    with open(file_path, "rb") as file:
      header_size = int.from_bytes(file.read(8), "little", signed=False)
      if header_size <= 0:
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
# Reading the connected LoRA loader's live enabled/disabled state.
#
# We don't import the loader's class - we look it up through ComfyUI's own shared
# node registry, then call its own `get_enabled_loras_from_prompt_node` classmethod
# if it has one (true reuse of its logic). If some other/older loader is connected
# without that classmethod, we fall back to reading its `lora_*` inputs ourselves,
# using the same field names ("on"/"lora") that convention uses.
# ---------------------------------------------------------------------------

def _get_enabled_loras_fallback(prompt_node):
  enabled = []
  for name, value in prompt_node.get("inputs", {}).items():
    if not name.startswith("lora_") or not isinstance(value, dict):
      continue
    if value.get("on") and value.get("lora"):
      enabled.append({"name": value["lora"]})
  return enabled


def get_enabled_loras(prompt_node):
  loader_class = nodes.NODE_CLASS_MAPPINGS.get(MXD_LORA_LOADER_CLASS_TYPE)
  get_enabled = getattr(loader_class, "get_enabled_loras_from_prompt_node", None)
  if callable(get_enabled):
    return get_enabled(prompt_node)
  return _get_enabled_loras_fallback(prompt_node)


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

    mxd_prompt_node = self._find_connected_loader_node(prompt, unique_id)
    enabled_loras = []
    if mxd_prompt_node is not None:
      enabled_loras = get_enabled_loras(mxd_prompt_node)
    elif prompt is not None:
      _log("`model` isn't wired directly from a Lora Loader MXD node; no triggers resolved.")

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
      lora_name = lora_entry["name"]
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
  def _find_connected_loader_node(prompt, unique_id):
    """Follows this node's own `model` input link back to the prompt-dict entry for
    the connected LoRA loader node, or None if not directly wired to one."""
    if not prompt or unique_id is None:
      return None
    this_node = prompt.get(str(unique_id))
    if this_node is None:
      return None
    model_input = this_node.get("inputs", {}).get("model")
    if not (isinstance(model_input, list) and len(model_input) == 2):
      return None
    upstream_node = prompt.get(str(model_input[0]))
    if upstream_node is None or upstream_node.get("class_type") != MXD_LORA_LOADER_CLASS_TYPE:
      return None
    return upstream_node


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


def _sha256_of_file(file_path):
  sha256 = hashlib.sha256()
  with open(file_path, "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 128), b""):
      sha256.update(chunk)
  return sha256.hexdigest()


@routes.get("/loratriggertap/api/resolve")
async def api_resolve_triggers(request):
  """Refresh: tiers 1-3 only, local reads, no network call."""
  files_param = _get_param(request, "files")
  files = [f for f in (files_param.split(",") if files_param else []) if f]
  data = {}
  for file in files:
    tier, words = get_trigger_word_tiers(file)
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
    file_hash = _sha256_of_file(file_path)
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
    except aiohttp.ClientError as exc:
      return web.json_response({"status": 502, "error": str(exc)})

  trained_words = [w.strip() for w in (data.get("trainedWords") or []) if isinstance(w, str) and w.strip()]

  sidecar_path = cm_info_sidecar_path(file_path)
  cm_info = _load_json_file(sidecar_path, default={}) or {}
  cm_info["TrainedWords"] = trained_words
  cm_info["civitaiModelId"] = data.get("modelId")
  cm_info["civitaiModelVersionId"] = data.get("id")
  cm_info["fetchedAt"] = time.time()
  _save_json_file(sidecar_path, cm_info)

  return web.json_response({"status": 200, "data": {"words": trained_words}})
