import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(
  new URL("../web/lora_trigger_tap.js", import.meta.url),
  "utf8",
);
const helpersStart = source.indexOf("function cleanLoraName");
const helpersEnd = source.indexOf("function loaderEntries");
assert.notEqual(helpersStart, -1);
assert.notEqual(helpersEnd, -1);

const context = {};
vm.createContext(context);
vm.runInContext(
  `
    const MXD_LORA_LOADER_TYPE = "Lora Loader MXD";
    ${source.slice(helpersStart, helpersEnd)}
    directTuple = genericEntries({
      widgets: [{
        name: "lora_stack",
        value: ["direct.safetensors", 0.8, 0.8],
      }],
    });
    tupleList = genericEntries({
      widgets: [{
        name: "lora_stack",
        value: [["one.safetensors", 1, 1], ["two.safetensors", 0.5, 0.5]],
      }],
    });
  `,
  context,
);

assert.deepEqual(JSON.parse(JSON.stringify(context.directTuple)), [
  { name: "direct.safetensors", strength: 0.8, enabled: true },
]);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.tupleList)).map((entry) => entry.name),
  ["one.safetensors", "two.safetensors"],
);
assert.match(source, /params\.append\("file", file\)/);
assert.doesNotMatch(source, /files\.join\(","\)/);
assert.match(source, /this\.triggerDraw\?\.\(\)/);

console.log("Frontend compatibility regressions: OK");
