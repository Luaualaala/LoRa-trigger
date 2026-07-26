import asyncio
import importlib.util
import pathlib
import sys
import types
import unittest


class _Routes:
  def get(self, _path):
    return lambda function: function


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_full_path = lambda _kind, _name: None
nodes = types.ModuleType("nodes")
nodes.NODE_CLASS_MAPPINGS = {}
aiohttp = types.ModuleType("aiohttp")
aiohttp.ClientError = Exception
aiohttp.ClientSession = object
aiohttp.ClientTimeout = object
aiohttp_web = types.ModuleType("aiohttp.web")
aiohttp_web.json_response = lambda value: value
aiohttp.web = aiohttp_web
server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
sys.modules.update({
  "aiohttp": aiohttp,
  "aiohttp.web": aiohttp_web,
  "folder_paths": folder_paths,
  "nodes": nodes,
  "server": server,
})

MODULE_PATH = pathlib.Path(__file__).parents[1] / "lora_trigger_tap.py"
SPEC = importlib.util.spec_from_file_location("lora_trigger_tap_under_test", MODULE_PATH)
tap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tap)


class LoaderCompatibilityTests(unittest.TestCase):
  def test_mxd_uses_its_own_extractor_first(self):
    class Mxd:
      @classmethod
      def get_enabled_loras_from_prompt_node(cls, _node):
        return [{"name": "mxd.safetensors"}]

    nodes.NODE_CLASS_MAPPINGS[tap.MXD_LORA_LOADER_CLASS_TYPE] = Mxd
    prompt_node = {
      "class_type": tap.MXD_LORA_LOADER_CLASS_TYPE,
      "inputs": {"lora_1": {"on": False, "lora": "wrong.safetensors"}},
    }
    self.assertEqual(tap.get_enabled_loras(prompt_node), [{"name": "mxd.safetensors"}])

  def test_mxd_extractor_failure_falls_back(self):
    class BrokenMxd:
      @classmethod
      def get_enabled_loras_from_prompt_node(cls, _node):
        raise AttributeError("changed schema")

    previous = nodes.NODE_CLASS_MAPPINGS.get(tap.MXD_LORA_LOADER_CLASS_TYPE)
    nodes.NODE_CLASS_MAPPINGS[tap.MXD_LORA_LOADER_CLASS_TYPE] = BrokenMxd
    self.addCleanup(
      lambda: nodes.NODE_CLASS_MAPPINGS.__setitem__(
        tap.MXD_LORA_LOADER_CLASS_TYPE, previous
      ) if previous is not None else nodes.NODE_CLASS_MAPPINGS.pop(
        tap.MXD_LORA_LOADER_CLASS_TYPE, None
      )
    )
    prompt_node = {
      "class_type": tap.MXD_LORA_LOADER_CLASS_TYPE,
      "inputs": {"lora_1": {"on": True, "lora": "fallback.safetensors"}},
    }
    self.assertEqual(
      tap.get_enabled_loras(prompt_node),
      [{"name": "fallback.safetensors"}],
    )

  def test_builtin_and_zero_strength(self):
    enabled = {
      "class_type": "LoraLoader",
      "inputs": {
        "lora_name": "character.safetensors",
        "strength_model": 0.8,
        "strength_clip": 0.5,
      },
    }
    disabled = {
      "class_type": "LoraLoader",
      "inputs": {
        "lora_name": "disabled.safetensors",
        "strength_model": 0,
        "strength_clip": 0,
      },
    }
    self.assertEqual(tap.get_enabled_loras(enabled)[0]["name"], "character.safetensors")
    self.assertEqual(tap.get_enabled_loras(disabled), [])

  def test_dynamic_and_stack_formats(self):
    node = {
      "class_type": "ThirdPartyPowerLoader",
      "inputs": {
        "lora_1": {"on": True, "lora": "a.safetensors", "strength": 1},
        "lora_2": {"enabled": False, "lora_name": "off.safetensors"},
        "lora_stack": [
          {"active": True, "filename": "b.safetensors", "weight": 0.7},
          ["c.safetensors", 0.5, 0.5],
        ],
      },
    }
    self.assertEqual(
      [entry["name"] for entry in tap.get_enabled_loras(node)],
      ["a.safetensors", "b.safetensors", "c.safetensors"],
    )

  def test_upstream_chain_order_and_passthrough(self):
    prompt = {
      "1": {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "first.safetensors", "strength_model": 1},
      },
      "2": {
        "class_type": "PassThrough",
        "inputs": {"model": ["1", 0]},
      },
      "3": {
        "class_type": "LoraLoader",
        "inputs": {
          "model": ["2", 0],
          "lora_name": "second.safetensors",
          "strength_model": 1,
        },
      },
      "4": {
        "class_type": tap.NODE_NAME,
        "inputs": {"model": ["3", 0]},
      },
    }
    found = tap.MxdLoraTriggerTap._find_connected_loader_nodes(prompt, "4")
    self.assertEqual(
      [tap.get_enabled_loras(node)[0]["name"] for node in found],
      ["first.safetensors", "second.safetensors"],
    )

  def test_separate_lora_stack_provider_is_traversed(self):
    prompt = {
      "1": {
        "class_type": "LoraStackProvider",
        "inputs": {"lora_stack": [["stacked.safetensors", 0.6, 0.6]]},
      },
      "2": {
        "class_type": "StackApplyingLoader",
        "inputs": {"lora_stack": ["1", 0]},
      },
      "3": {
        "class_type": tap.NODE_NAME,
        "inputs": {"model": ["2", 0]},
      },
    }
    found = tap.MxdLoraTriggerTap._find_connected_loader_nodes(prompt, "3")
    self.assertEqual(len(found), 1)
    self.assertEqual(tap.get_enabled_loras(found[0])[0]["name"], "stacked.safetensors")

  def test_lora_stack_graph_link_is_not_a_lora_tuple(self):
    prompt = {
      "provider-uuid": {"class_type": "StackProvider", "inputs": {}},
      "consumer": {
        "class_type": "StackConsumer",
        "inputs": {"lora_stack": ["provider-uuid", 1]},
      },
    }
    self.assertEqual(
      tap.get_enabled_loras(prompt["consumer"], prompt=prompt),
      [],
    )

  def test_string_trigger_metadata_stays_one_word(self):
    self.assertEqual(tap._normalize_trigger_words("solo_trigger"), ["solo_trigger"])
    previous = tap._load_json_file
    tap._load_json_file = lambda *_args, **_kwargs: {"TrainedWords": "solo_trigger"}
    self.addCleanup(lambda: setattr(tap, "_load_json_file", previous))
    self.assertEqual(
      tap.read_cm_info_sidecar("fake.safetensors"),
      ["solo_trigger"],
    )

  def test_resolve_route_preserves_commas_in_filename(self):
    class Query(dict):
      def getall(self, key, default=None):
        return self.get(key, default or [])

    request = types.SimpleNamespace(
      rel_url=types.SimpleNamespace(
        query=Query({"file": ["portrait, final.safetensors", "style.safetensors"]})
      )
    )
    previous = tap.get_trigger_word_tiers
    tap.get_trigger_word_tiers = lambda name: (1, [name])
    self.addCleanup(lambda: setattr(tap, "get_trigger_word_tiers", previous))
    result = asyncio.run(tap.api_resolve_triggers(request))
    self.assertEqual(
      list(result["data"]),
      ["portrait, final.safetensors", "style.safetensors"],
    )

  def test_civitai_timeout_returns_json_error(self):
    class TimeoutRequest:
      rel_url = types.SimpleNamespace(query={"file": "timeout.safetensors"})

    class TimeoutResponse:
      async def __aenter__(self):
        raise asyncio.TimeoutError("timed out")

      async def __aexit__(self, *_args):
        return False

    class TimeoutSession:
      async def __aenter__(self):
        return self

      async def __aexit__(self, *_args):
        return False

      def get(self, *_args, **_kwargs):
        return TimeoutResponse()

    previous_path = tap.get_lora_file_path
    previous_hash = tap._sha256_of_file
    previous_session = tap.aiohttp.ClientSession
    tap.get_lora_file_path = lambda _name: "timeout.safetensors"
    tap._sha256_of_file = lambda _path: "abc123"
    tap.aiohttp.ClientSession = TimeoutSession
    self.addCleanup(lambda: setattr(tap, "get_lora_file_path", previous_path))
    self.addCleanup(lambda: setattr(tap, "_sha256_of_file", previous_hash))
    self.addCleanup(lambda: setattr(tap.aiohttp, "ClientSession", previous_session))
    result = asyncio.run(tap.api_fetch_civitai(TimeoutRequest()))
    self.assertEqual(result["status"], 502)

  def test_backend_execution_does_not_require_frontend_widgets(self):
    prompt = {
      "1": {
        "class_type": "LoraLoader",
        "inputs": {
          "lora_name": "desktop.safetensors",
          "strength_model": 1,
          "strength_clip": 1,
        },
      },
      "2": {
        "class_type": tap.NODE_NAME,
        "inputs": {"model": ["1", 0]},
      },
    }
    model = object()
    clip = object()
    result = tap.MxdLoraTriggerTap().get_triggers(
      model=model,
      clip=clip,
      prompt=prompt,
      unique_id="2",
    )
    self.assertIs(result[0], model)
    self.assertIs(result[1], clip)
    self.assertEqual(result[2], "")
    self.assertEqual(
      result[3],
      '[{"name": "desktop.safetensors", "category": "none", "triggers": []}]',
    )


if __name__ == "__main__":
  unittest.main()
