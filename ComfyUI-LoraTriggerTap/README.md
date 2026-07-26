# ComfyUI-LoraTriggerTap

Companion node for rgthree-style "Power LoRA Loader" nodes (built and tested against
[Maxed-Out-99/ComfyUI-MaxedOut](https://github.com/Maxed-Out-99/ComfyUI-MaxedOut)'s
`Lora Loader MXD`). Wire it right after your loader and it taps the trigger words for
whichever LoRAs are currently enabled onto a `triggers` STRING output, so they follow
each LoRA's own on/off state instead of being copy-pasted by hand.

```
Lora Loader MXD -> LoRA Trigger Tap -> your CLIPTextEncode / prompt node
```

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
- **Requires a "Lora Loader MXD"-compatible node to also be installed** - this node
  only reads that node's live enabled/disabled state; it doesn't load LoRAs itself.
  If none is found upstream, it logs a warning and outputs an empty string rather
  than failing.
- Trigger words are resolved locally (no network) from, in order: a `.cm-info.json`
  sidecar next to the `.safetensors`, then `modelspec.trigger_phrase`, then
  `caption_prefix` in the file's own safetensors header. A per-LoRA "Fetch" button
  is offered only when none of those have anything, and calls CivitAI's public
  hash-lookup API on demand (never automatically), caching the result into the same
  `.cm-info.json` sidecar.
- Click **Refresh** any time to resync the LoRA list and re-resolve triggers for
  newly-added LoRAs without re-running the graph.
