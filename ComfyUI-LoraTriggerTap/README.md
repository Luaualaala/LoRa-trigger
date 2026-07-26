# ComfyUI-LoraTriggerTap

Trigger-word companion for ComfyUI LoRA loaders. `Lora Loader MXD` from
[Maxed-Out-99/ComfyUI-MaxedOut](https://github.com/Maxed-Out-99/ComfyUI-MaxedOut)
is a first-class integration and remains the primary supported loader. A universal
compatibility layer also supports ComfyUI's built-in Load LoRA, chained loaders,
rgthree-style power loaders, stack/list loaders, and loaders that expose common
numbered LoRA fields.

```
Lora Loader MXD -> LoRA Trigger Tap -> your CLIPTextEncode / prompt node
```

You can also connect the tap after one or more built-in or third-party loaders.
It follows upstream MODEL and CLIP links, including through pass-through nodes.

## ComfyUI Desktop

The node runs in ComfyUI Desktop's bundled Python server, not only in the client
interface. Trigger extraction, MODEL/CLIP pass-through, local metadata lookup, and
`lora_info` generation all work at backend execution time and do not require the
JavaScript panel to be present. The interface uses ComfyUI's own API client so its
local routes work through Desktop's embedded server.

Install the complete folder in the user-selected `custom_nodes` directory shown by
ComfyUI Desktop; do not place it in Desktop's managed `resource/ComfyUI` directory,
which is replaced during application updates. Restart ComfyUI after installation.

The rich per-LoRA controls support both classic LiteGraph rendering and the
Vue-based Nodes 2.0 `WidgetLegacy` host. They isolate Vue-owned widget sizing,
request redraws through Nodes 2.0's widget hook, and retain the classic canvas
redraw path. Notifications use Desktop's native ComfyUI toast API when available.

- Inputs: `model`, `clip` (from the loader). Outputs: `model`, `clip` (unchanged
  pass-through), `triggers` (STRING), `lora_info` (STRING, JSON array of
  `{name, category, triggers}` — one entry per currently-enabled LoRA).
- Each LoRA row has a small category badge (top-right, next to the name) —
  click it to cycle **— / Char / Style / Pose / Cloth**. Purely informational
  here; it exists so a downstream tool (e.g. Prompt Forge) can know which
  prompt field each LoRA's trigger words belong to, without guessing from the
  filename. Defaults to `—` (none) and is saved with the workflow.
  Prompt Forge specifically doesn't read this output at all — it reads this
  node's widgets directly from the graph (same decoupled, no-import pattern
  this node uses to read the LoRA loader), so detection works whether or not
  `lora_info`/`triggers` are wired anywhere.
- This node doesn't load LoRAs itself. MXD uses its own registered enabled-LoRA
  extractor as the source of truth. Other loaders are read through a dependency-free
  adapter for standard `lora_name` inputs, dynamic LoRA records, stack/list records,
  and common numbered fields. Unknown loaders fail safely with an empty trigger list.
- Trigger words are resolved locally (no network) from, in order: a `.cm-info.json`
  sidecar next to the `.safetensors`, then `modelspec.trigger_phrase`, then
  `caption_prefix` in the file's own safetensors header. A per-LoRA "Fetch" button
  is offered only when none of those have anything, and calls CivitAI's public
  hash-lookup API on demand (never automatically), caching the result into the same
  `.cm-info.json` sidecar.
- Click **Refresh** any time to resync the LoRA list and re-resolve triggers for
  newly-added LoRAs without re-running the graph.
