import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Self-contained on purpose: no imports from any other custom-node plugin's web/
// folder, so this file has zero cross-plugin dependencies (relative imports across
// plugin boundaries aren't a robust mechanism in ComfyUI's extension loading).

const NODE_TYPE = "LoRA Trigger Tap";
const MXD_LORA_LOADER_TYPE = "Lora Loader MXD";
const ROW_HEIGHT = LiteGraph.NODE_WIDGET_HEIGHT || 20;
const API_BASE = "/loratriggertap/api";

// Per-LoRA category, click-to-cycle on the badge next to each LoRA's name.
// Consumed by downstream nodes (e.g. Prompt Forge) that route a LoRA's trigger
// words into the matching prompt field instead of one undifferentiated string.
const LORA_CATEGORIES = ["none", "character", "style", "pose", "clothes"];
const LORA_CATEGORY_LABELS = { none: "—", character: "Char", style: "Style", pose: "Pose", clothes: "Cloth" };
const LORA_CATEGORY_BADGE_WIDTH = 54;

function nextLoraCategory(current) {
  const i = LORA_CATEGORIES.indexOf(current);
  return LORA_CATEGORIES[(i + 1) % LORA_CATEGORIES.length];
}

// ---------------------------------------------------------------------------
// Small canvas/array helpers (duplicated in miniature rather than shared, to keep
// this a single, dependency-free file).
// ---------------------------------------------------------------------------

function moveArrayItem(arr, itemOrFrom, to) {
  const from = typeof itemOrFrom === "number" ? itemOrFrom : arr.indexOf(itemOrFrom);
  arr.splice(to, 0, arr.splice(from, 1)[0]);
}

function removeArrayItem(arr, itemOrIndex) {
  const index = typeof itemOrIndex === "number" ? itemOrIndex : arr.indexOf(itemOrIndex);
  arr.splice(index, 1);
}

function isLowQuality() {
  return (app.canvas?.ds?.scale || 1) <= 0.5;
}

function measureText(ctx, str) {
  return ctx.measureText(str).width;
}

function fitString(ctx, str, maxWidth) {
  const width = measureText(ctx, str);
  const ellipsis = "…";
  const ellipsisWidth = measureText(ctx, ellipsis);
  if (width <= maxWidth || width <= ellipsisWidth) return str;
  let lo = 0;
  let hi = str.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measureText(ctx, str.slice(0, mid)) <= maxWidth - ellipsisWidth) lo = mid;
    else hi = mid - 1;
  }
  return str.slice(0, lo) + ellipsis;
}

function drawRoundedRectangle(ctx, { pos, size, borderRadius, colorBackground, colorStroke }) {
  const lowQuality = isLowQuality();
  ctx.save();
  ctx.strokeStyle = colorStroke || LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.fillStyle = colorBackground || LiteGraph.WIDGET_BGCOLOR;
  ctx.beginPath();
  ctx.roundRect(pos[0], pos[1], size[0], size[1], lowQuality ? [0] : [borderRadius ?? size[1] * 0.5]);
  ctx.fill();
  if (!lowQuality) ctx.stroke();
  ctx.restore();
}

function drawTogglePart(ctx, { posX, posY, height, value }) {
  const lowQuality = isLowQuality();
  const alpha = app.canvas?.editor_alpha ?? 1;
  ctx.save();
  const toggleRadius = height * 0.36;
  const toggleBgWidth = height * 1.5;
  if (!lowQuality) {
    ctx.beginPath();
    ctx.roundRect(posX + 4, posY + 4, toggleBgWidth - 8, height - 8, [height * 0.5]);
    ctx.globalAlpha = alpha * 0.25;
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fill();
    ctx.globalAlpha = alpha;
  }
  ctx.fillStyle = value === true ? "#89B" : "#888";
  const toggleX = lowQuality || value === false ? posX + height * 0.5 : posX + height;
  ctx.beginPath();
  ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  return [posX, toggleBgWidth];
}

async function apiResolve(files) {
  const params = new URLSearchParams();
  for (const file of files) params.append("file", file);
  const r = await api.fetchApi(`${API_BASE}/resolve?${params.toString()}`, {
    cache: "no-store",
  });
  return r.json();
}

async function apiFetchCivitai(file) {
  const r = await api.fetchApi(`${API_BASE}/fetch_civitai?file=${encodeURIComponent(file)}`, {
    cache: "no-store",
  });
  return r.json();
}

function showMessage(message) {
  console.error(`[LoRA Trigger Tap] ${message}`);
  const toast = app.extensionManager?.toast;
  if (toast?.add) {
    toast.add({
      severity: "error",
      summary: "LoRA Trigger Tap",
      detail: message.replace(/^LoRA Trigger Tap:\s*/, ""),
      life: 4000,
    });
    return;
  }

  // Compatibility fallback for older ComfyUI frontends without the toast API.
  let container = document.querySelector(".loratriggertap-messages");
  if (!container) {
    container = document.createElement("div");
    container.className = "loratriggertap-messages";
    container.style.cssText =
      "position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;flex-direction:column;gap:4px;align-items:center;";
    document.body.appendChild(container);
  }
  const node = document.createElement("div");
  node.textContent = message;
  node.style.cssText =
    "background:#552222;color:#fff;padding:6px 12px;border-radius:4px;font:12px sans-serif;box-shadow:0 2px 6px rgba(0,0,0,0.4);";
  container.appendChild(node);
  setTimeout(() => node.remove(), 4000);
}

// ---------------------------------------------------------------------------
// Loader compatibility. MXD is intentionally the first adapter; generic loaders
// are normalized from their widgets without importing any other extension.
// ---------------------------------------------------------------------------

function cleanLoraName(value) {
  if (typeof value !== "string") return null;
  const name = value.trim();
  return name && !["none", "null", "undefined"].includes(name.toLowerCase()) ? name : null;
}

function strengthFromObject(value) {
  for (const key of ["strength", "strength_model", "strength_clip", "model_strength", "clip_strength", "weight"]) {
    if (typeof value?.[key] === "number") return value[key];
  }
  return undefined;
}

function normalizeObjectEntry(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const name = ["lora", "lora_name", "filename", "file", "name"]
    .map((key) => cleanLoraName(value[key]))
    .find(Boolean);
  if (!name) return null;
  const strength = strengthFromObject(value);
  const enabled =
    !["on", "enabled", "active"].some((key) => key in value && !value[key]) &&
    value.bypass !== true &&
    value.muted !== true &&
    strength !== 0;
  return { name, strength, enabled };
}

function normalizeArrayEntry(value) {
  if (!Array.isArray(value) || !value.length) return null;
  const name = cleanLoraName(value[0]);
  if (!name) return null;
  const strengths = value.slice(1, 3).filter((item) => typeof item === "number");
  const enabled = !strengths.length || strengths.some((item) => item !== 0);
  return { name, strength: strengths[0], enabled };
}

function isLoaderPathInput(name) {
  const normalized = String(name || "").toLowerCase();
  return ["model", "clip", "lora"].some((token) => normalized.includes(token));
}

function mxdEntries(node) {
  if (node?.type !== MXD_LORA_LOADER_TYPE) return null;
  return (node.widgets || [])
    .filter((widget) => widget.name?.startsWith("lora_"))
    .map((widget) => normalizeObjectEntry(widget.value))
    .filter(Boolean);
}

function genericEntries(node) {
  const widgets = node?.widgets || [];
  const entries = [];
  const singleNameWidget = widgets.find((widget) => widget.name === "lora_name");
  const singleName = cleanLoraName(singleNameWidget?.value);
  if (singleName) {
    const modelStrength = widgets.find((widget) => widget.name === "strength_model")?.value;
    const clipStrength = widgets.find((widget) => widget.name === "strength_clip")?.value;
    if (modelStrength !== 0 || clipStrength !== 0) {
      entries.push({ name: singleName, strength: modelStrength ?? clipStrength, enabled: true });
    } else {
      entries.push({ name: singleName, strength: 0, enabled: false });
    }
  }

  for (const widget of widgets) {
    const key = String(widget.name || "").toLowerCase();
    if (widget === singleNameWidget) continue;
    const objectEntry = normalizeObjectEntry(widget.value);
    if (objectEntry && (key.includes("lora") || "lora" in widget.value || "lora_name" in widget.value)) {
      entries.push(objectEntry);
      continue;
    }
    if (typeof widget.value === "string" && key.includes("lora") &&
        (key.includes("name") || key.includes("file") || key.startsWith("lora_"))) {
      const name = cleanLoraName(widget.value);
      if (name) entries.push({ name, enabled: true });
    }
    if (Array.isArray(widget.value) && key.includes("lora")) {
      const directTuple = normalizeArrayEntry(widget.value);
      if (directTuple) {
        entries.push(directTuple);
      } else {
        for (const item of widget.value) {
          const entry = normalizeObjectEntry(item) || normalizeArrayEntry(item);
          if (entry) entries.push(entry);
        }
      }
    }
  }
  return [...new Map(entries.map((entry) => [entry.name, entry])).values()];
}

function loaderEntries(node) {
  // First-class MXD adapter always gets first refusal.
  const primary = mxdEntries(node);
  return primary !== null ? primary : genericEntries(node);
}

// ---------------------------------------------------------------------------
// A minimal "custom" LiteGraph widget base: rebuilds a list of clickable regions
// every draw() call and dispatches pointerdown hits to them. Simpler than a full
// hitAreas system (no drag tracking) - all this node needs is click-to-toggle.
// ---------------------------------------------------------------------------

class ClickRegionWidget {
  constructor(name) {
    this.name = name;
    this.type = "custom";
    this._clickRegions = [];

    // Nodes 2.0 renders legacy custom widgets inside a Vue WidgetLegacy host.
    // That host owns its canvas width and may write it back onto the shared widget,
    // which breaks LiteGraph hit-testing. These widgets always draw at the width
    // passed to draw(), so external width writes are intentionally ignored.
    Object.defineProperty(this, "width", {
      configurable: true,
      get: () => undefined,
      set: () => {},
    });
  }

  addClickRegion(x, y, w, h, onClick) {
    this._clickRegions.push({ x, y, w, h, onClick });
  }

  mouse(event, pos, node) {
    if (!["pointerdown", "mousedown"].includes(event.type)) return false;
    for (const region of this._clickRegions) {
      if (
        pos[0] >= region.x &&
        pos[0] <= region.x + region.w &&
        pos[1] >= region.y &&
        pos[1] <= region.y + region.h
      ) {
        region.onClick(event, pos, node);
        this.triggerRedraw(node);
        return true;
      }
    }
    return false;
  }

  triggerRedraw(node) {
    // WidgetLegacy attaches triggerDraw in Nodes 2.0. The other calls retain
    // compatibility with the classic LiteGraph renderer and older frontends.
    this.triggerDraw?.();
    node?.setDirtyCanvas?.(true, false);
    app.canvas?.setDirty?.(true, false);
  }
}

class RefreshButtonWidget extends ClickRegionWidget {
  constructor() {
    super("refresh_button");
    this.value = "";
    this.serializeValue = () => undefined;
  }

  computeSize(width) {
    return [width, ROW_HEIGHT];
  }

  draw(ctx, node, width, posY, height) {
    this._clickRegions = [];
    const margin = 15;
    drawRoundedRectangle(ctx, {
      pos: [margin, posY],
      size: [width - margin * 2, height],
      colorBackground: LiteGraph.WIDGET_BGCOLOR,
    });
    if (isLowQuality()) return;
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.fillText("Refresh", width / 2, posY + height / 2);
    ctx.restore();
    this.addClickRegion(margin, posY, width - margin * 2, height, (event, pos, node) => node.doRefresh());
  }
}

const DEFAULT_TRIGGER_WIDGET_DATA = { lora: null, words: [], category: "none" };

class TriggerLoraGroupWidget extends ClickRegionWidget {
  constructor(name) {
    super(name);
    this.fetching = false;
    this._value = { ...DEFAULT_TRIGGER_WIDGET_DATA };
  }

  set value(v) {
    this._value = typeof v === "object" && v !== null ? v : { ...DEFAULT_TRIGGER_WIDGET_DATA };
    if (!Array.isArray(this._value.words)) this._value.words = [];
    if (!LORA_CATEGORIES.includes(this._value.category)) this._value.category = "none";
  }

  get value() {
    return this._value;
  }

  computeSize(width) {
    const rows = this.value.words.length || 1; // 1 for "no data found" + Fetch row
    return [width, ROW_HEIGHT + rows * ROW_HEIGHT + 6];
  }

  getStrength(node) {
    const entry = node.getConnectedLoraEntries?.().find((item) => item.name === this.value.lora);
    return entry?.strength;
  }

  draw(ctx, node, width, posY, height) {
    this._clickRegions = [];
    const margin = 10;
    const innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();

    ctx.save();

    let rowY = posY + 2;
    const headerMidY = rowY + ROW_HEIGHT * 0.5;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const strength = this.getStrength(node);
    const label =
      strength != null ? `${this.value.lora} (${Number(strength).toFixed(2)})` : `${this.value.lora}`;
    const badgeWidth = lowQuality ? 0 : LORA_CATEGORY_BADGE_WIDTH;
    ctx.fillText(fitString(ctx, label, width - margin * 2 - badgeWidth), margin, headerMidY);

    if (!lowQuality) {
      const category = this.value.category || "none";
      const badgeX = width - margin - badgeWidth;
      drawRoundedRectangle(ctx, {
        pos: [badgeX, rowY + 1],
        size: [badgeWidth, ROW_HEIGHT - 6],
        borderRadius: 4,
        colorBackground: category === "none" ? LiteGraph.WIDGET_BGCOLOR : "#2e4a46",
        colorStroke: category === "none" ? undefined : "#5fb8ad",
      });
      ctx.save();
      ctx.textAlign = "center";
      ctx.fillStyle = category === "none" ? LiteGraph.WIDGET_TEXT_COLOR : "#8fd4c9";
      ctx.fillText(LORA_CATEGORY_LABELS[category] || category, badgeX + badgeWidth / 2, headerMidY);
      ctx.restore();
      ctx.textAlign = "left";
      this.addClickRegion(badgeX, rowY, badgeWidth, ROW_HEIGHT - 4, () => {
        this.value.category = nextLoraCategory(category);
        node.setDirtyCanvas(true, true);
      });
    }
    rowY += ROW_HEIGHT;

    if (lowQuality) {
      ctx.restore();
      return;
    }

    const words = this.value.words || [];
    if (!words.length) {
      const midY = rowY + ROW_HEIGHT * 0.5;
      ctx.save();
      ctx.globalAlpha = (app.canvas?.editor_alpha ?? 1) * 0.6;
      ctx.fillText(this.fetching ? "Fetching…" : "(no local data found)", margin + innerMargin, midY);
      ctx.restore();

      if (!this.fetching) {
        const btnWidth = 60;
        const btnX = width - margin - btnWidth;
        drawRoundedRectangle(ctx, {
          pos: [btnX, rowY + 2],
          size: [btnWidth, ROW_HEIGHT - 4],
          borderRadius: 4,
          colorBackground: LiteGraph.WIDGET_BGCOLOR,
        });
        ctx.textAlign = "center";
        ctx.fillText("Fetch", btnX + btnWidth / 2, midY);
        ctx.textAlign = "left";
        this.addClickRegion(btnX, rowY, btnWidth, ROW_HEIGHT, (event, pos, node) =>
          node.fetchTriggerWordsFor(this),
        );
      }
      rowY += ROW_HEIGHT;
    } else {
      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const midY = rowY + ROW_HEIGHT * 0.5;
        const toggleBounds = drawTogglePart(ctx, {
          posX: margin + innerMargin,
          posY: rowY,
          height: ROW_HEIGHT,
          value: !!word.checked,
        });
        const textX = margin + innerMargin + toggleBounds[1] + innerMargin;
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.fillText(
          fitString(ctx, word.word, width - margin * 2 - toggleBounds[1] - innerMargin * 2),
          textX,
          midY,
        );
        this.addClickRegion(margin + innerMargin, rowY, width - margin * 2 - innerMargin, ROW_HEIGHT, () => {
          word.checked = !word.checked;
          node.setDirtyCanvas(true, true);
        });
        rowY += ROW_HEIGHT;
      }
    }

    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// Node behavior, attached via prototype patching in beforeRegisterNodeDef - the
// standard ComfyUI extension pattern for customizing a server-defined node's UI
// without needing a shared base class from another plugin.
// ---------------------------------------------------------------------------

function addNewTriggerWidget(node) {
  node.triggerWidgetsCounter = (node.triggerWidgetsCounter || 0) + 1;
  return node.addCustomWidget(new TriggerLoraGroupWidget("trigger_" + node.triggerWidgetsCounter));
}

function addNonTriggerWidgets(node) {
  node.refreshButtonWidget = node.addCustomWidget(new RefreshButtonWidget());
}

function recomputeSize(node) {
  const computed = node.computeSize();
  node.size = node.size || [0, 0];
  node.size[0] = Math.max(node.size[0], computed[0]);
  node.size[1] = Math.max(node.size[1], computed[1]);
}

function getTriggerWidgets(node) {
  return node.widgets.filter((w) => w.name?.startsWith("trigger_"));
}

function redrawCustomWidgets(node) {
  for (const widget of node.widgets || []) {
    widget.triggerRedraw?.(node);
  }
}

app.registerExtension({
  name: "loratriggertap.LoraTriggerTap",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      if (!this.refreshButtonWidget) {
        addNonTriggerWidgets(this);
      }
      recomputeSize(this);
      redrawCustomWidgets(this);
      this.setDirtyCanvas(true, true);
    };

    const onConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      // Clear directly rather than via this.removeWidget(0) in a loop - ComfyUI's
      // stock removeWidget throws "Widget not found on this node" if anything else
      // in the load pipeline (e.g. another extension) has already touched this
      // node's widgets array by the time we get here. None of our custom widgets
      // have linked inputs/DOM state that needs removeWidget's cleanup anyway.
      if (this.widgets) this.widgets.length = 0;
      this.refreshButtonWidget = null;

      onConfigure?.apply(this, [info]);

      // Divider-free here (single button up top) so it stays pinned above the LoRA
      // groups - trigger widgets added below just append to the end.
      addNonTriggerWidgets(this);

      for (const widgetValue of info.widgets_values || []) {
        if (widgetValue?.lora !== undefined) {
          const widget = addNewTriggerWidget(this);
          widget.value = {
            lora: widgetValue.lora,
            words: widgetValue.words || [],
            category: widgetValue.category,
          };
        }
      }

      recomputeSize(this);
      redrawCustomWidgets(this);
      this.setDirtyCanvas(true, true);
    };

    nodeType.prototype.findConnectedLoaderNodes = function () {
      const found = [];
      const visited = new Set();
      const visit = (upstreamNode) => {
        if (!upstreamNode || visited.has(upstreamNode.id)) return;
        visited.add(upstreamNode.id);
        for (const input of upstreamNode.inputs || []) {
          if (!isLoaderPathInput(input.name) || input.link == null) continue;
          const linkInfo = upstreamNode.graph?.links?.[input.link];
          visit(linkInfo ? upstreamNode.graph.getNodeById(linkInfo.origin_id) : null);
        }
        if (loaderEntries(upstreamNode).length) found.push(upstreamNode);
      };
      for (const input of this.inputs || []) {
        if (!isLoaderPathInput(input.name) || input.link == null) continue;
        const linkInfo = this.graph?.links?.[input.link];
        visit(linkInfo ? this.graph.getNodeById(linkInfo.origin_id) : null);
      }
      return found;
    };

    nodeType.prototype.findConnectedLoaderNode = function () {
      const loaders = this.findConnectedLoaderNodes();
      return loaders.find((node) => node.type === MXD_LORA_LOADER_TYPE) || loaders.at(-1) || null;
    };

    nodeType.prototype.getConnectedLoraEntries = function () {
      const entries = this.findConnectedLoaderNodes().flatMap((node) => loaderEntries(node));
      return [...new Map(entries.map((entry) => [entry.name, entry])).values()];
    };

    nodeType.prototype.doRefresh = async function () {
      const loraEntries = this.getConnectedLoraEntries();
      if (!loraEntries.length) {
        showMessage("LoRA Trigger Tap: connect MODEL or CLIP from a compatible LoRA loader.");
        return;
      }

      const activeLoraFiles = loraEntries.map((entry) => entry.name);

      const existingByLora = new Map();
      for (const w of getTriggerWidgets(this)) {
        existingByLora.set(w.value.lora, w);
      }

      // Only resolve LoRAs we don't already have a trigger widget for - an existing
      // widget's checkbox picks (or its "no local data" state) are left alone so a
      // disabled-in-loader LoRA's prior picks survive being re-enabled later.
      const filesNeedingResolve = activeLoraFiles.filter((f) => !existingByLora.has(f));
      let resolved = {};
      if (filesNeedingResolve.length) {
        try {
          const res = await apiResolve(filesNeedingResolve);
          if (res?.status !== 200) {
            showMessage(`LoRA Trigger Tap: ${res?.error || "local trigger refresh failed"}.`);
            return;
          }
          resolved = res.data || {};
        } catch (error) {
          showMessage(`LoRA Trigger Tap: local trigger refresh failed (${error?.message || error}).`);
          return;
        }
      }

      // Drop trigger widgets for LoRAs the loader no longer lists at all (removed via
      // its own Remove, not just disabled - disabled ones stay in activeLoraFiles).
      for (const [loraFile, widget] of existingByLora) {
        if (!activeLoraFiles.includes(loraFile)) {
          removeArrayItem(this.widgets, widget);
          existingByLora.delete(loraFile);
        }
      }

      for (const loraEntry of loraEntries) {
        const loraFile = loraEntry.name;
        if (!existingByLora.has(loraFile)) {
          const info = resolved[loraFile];
          const words = (info?.words || []).map((word, i) => ({ word, checked: i === 0 }));
          const widget = addNewTriggerWidget(this);
          widget.value = { lora: loraFile, words };
          existingByLora.set(loraFile, widget);
        }
        // Move to the tail, in the loader's stack order - the Refresh button stays
        // pinned at the top since this only ever appends *after* it, never before.
        moveArrayItem(this.widgets, existingByLora.get(loraFile), this.widgets.length - 1);
      }

      recomputeSize(this);
      redrawCustomWidgets(this);
      this.setDirtyCanvas(true, true);
    };

    nodeType.prototype.fetchTriggerWordsFor = async function (triggerWidget) {
      if (triggerWidget.fetching) return;
      triggerWidget.fetching = true;
      triggerWidget.triggerRedraw(this);
      this.setDirtyCanvas(true, true);
      try {
        const res = await apiFetchCivitai(triggerWidget.value.lora);
        if (res.status === 429) {
          const retryAfter = res.retryAfter ? ` (retry after ${res.retryAfter}s)` : "";
          showMessage(`LoRA Trigger Tap: rate limited by CivitAI${retryAfter}.`);
          return;
        }
        if (res.status !== 200) {
          showMessage(`LoRA Trigger Tap: ${res.error || "CivitAI fetch failed"}.`);
          return;
        }
        const words = res.data?.words || [];
        triggerWidget.value.words = words.map((word, i) => ({ word, checked: i === 0 }));
      } catch (error) {
        showMessage(`LoRA Trigger Tap: CivitAI fetch failed (${error?.message || error}).`);
      } finally {
        triggerWidget.fetching = false;
        recomputeSize(this);
        triggerWidget.triggerRedraw(this);
        this.setDirtyCanvas(true, true);
      }
    };
  },
});
