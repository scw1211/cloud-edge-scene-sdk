"""用途：从配置或 Python entry point 动态加载、索引和卸载场景插件。"""

import importlib
import json
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from cloud_edge_framework.contracts import ContractError
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.plugins.base import ScenePlugin


DEFAULT_ENTRY_POINT_GROUP = "cloud_edge_framework.scenes"


class PluginLoadError(ValueError):
    """Raised when a plugin definition cannot be loaded without changing active state."""


@dataclass(frozen=True)
class PluginDescriptor:
    scene: str
    event_types: List[str]
    data_schema: str
    aliases: List[str]
    policy_version: str
    spec: str
    class_name: str
    health: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_options(options: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in options.items():
        if key.endswith("_path") and isinstance(value, str) and value:
            path = Path(value)
            resolved[key] = path if path.is_absolute() else project_root / path
        else:
            resolved[key] = value
    return resolved


def _instantiate(factory: Any, options: Dict[str, Any], source: str) -> ScenePlugin:
    try:
        candidate = factory(**options) if callable(factory) else factory
    except Exception as exc:  # noqa: BLE001
        raise PluginLoadError("failed to construct plugin {}: {}".format(source, exc)) from exc
    if not isinstance(candidate, ScenePlugin):
        raise PluginLoadError("plugin {} must create a ScenePlugin instance".format(source))
    return candidate


def load_plugin_spec(spec: str, options: Dict[str, Any], project_root: Path) -> ScenePlugin:
    if not isinstance(spec, str) or ":" not in spec:
        raise PluginLoadError("plugin spec must use module:attribute syntax")
    module_name, attribute_name = spec.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise PluginLoadError("plugin spec must use module:attribute syntax")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise PluginLoadError("cannot import plugin {}: {}".format(spec, exc)) from exc
    return _instantiate(factory, _resolve_options(options, project_root), spec)


def _entry_points(group: str) -> List[Any]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group))
    return list(discovered.get(group, []))


class SceneRegistry:
    def __init__(self, plugins: Optional[Iterable[ScenePlugin]] = None) -> None:
        self._plugins: Dict[str, ScenePlugin] = {}
        self._aliases: Dict[str, str] = {}
        self._event_types: Dict[str, str] = {}
        self._specs: Dict[str, str] = {}
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: ScenePlugin, spec: str = "python_instance") -> None:
        try:
            contract = plugin.contract_descriptor()
        except ContractError as exc:
            raise PluginLoadError("invalid plugin contract: {}".format(exc)) from exc
        scene = contract["scene"]
        event_types = list(contract["event_types"])
        if scene in self._plugins:
            raise PluginLoadError("scene is already registered: {}".format(scene))
        aliases = {str(alias).strip() for alias in plugin.aliases} | {scene}
        if "" in aliases:
            raise PluginLoadError("scene aliases must not be empty")
        conflicts = [alias for alias in aliases if alias in self._aliases]
        if conflicts:
            raise PluginLoadError("scene alias is already registered: {}".format(conflicts[0]))
        type_conflicts = [
            event_type
            for event_type in event_types
            if event_type in self._event_types
        ]
        if type_conflicts:
            raise PluginLoadError("event type is already registered: {}".format(type_conflicts[0]))
        self._plugins[scene] = plugin
        self._specs[scene] = spec
        for alias in aliases:
            self._aliases[alias] = scene
        for event_type in event_types:
            self._event_types[event_type] = scene

    def get(self, scene_or_alias: str) -> ScenePlugin:
        scene = self._aliases.get(str(scene_or_alias).strip())
        if scene is None:
            raise KeyError(
                "unsupported scene '{}'; loaded scenes: {}".format(
                    scene_or_alias, ", ".join(self.scenes()) or "none"
                )
            )
        return self._plugins[scene]

    def for_envelope(
        self,
        envelope: SceneEventEnvelope,
        validate: bool = True,
    ) -> ScenePlugin:
        plugin = self.get(envelope.scene)
        event_scene = self._event_types.get(envelope.event_type)
        if event_scene is None:
            raise KeyError("unsupported event type '{}'".format(envelope.event_type))
        if event_scene != plugin.scene:
            raise ContractError(
                "event type {!r} belongs to scene {!r}, not {!r}".format(
                    envelope.event_type, event_scene, plugin.scene
                )
            )
        if validate:
            plugin.validate_envelope(envelope)
        return plugin

    def for_payload(self, payload: Dict[str, Any]) -> ScenePlugin:
        return self.for_envelope(SceneEventEnvelope.from_dict(payload))

    def scenes(self) -> List[str]:
        return sorted(self._plugins)

    def descriptors(self) -> List[Dict[str, Any]]:
        descriptors = []
        for scene in self.scenes():
            plugin = self._plugins[scene]
            contract = plugin.contract_descriptor()
            descriptors.append(
                PluginDescriptor(
                    scene=scene,
                    event_types=list(contract["event_types"]),
                    data_schema=str(contract["data_schema"]),
                    aliases=sorted(set(plugin.aliases)),
                    policy_version=str(plugin.policy_version),
                    spec=self._specs[scene],
                    class_name="{}.{}".format(
                        plugin.__class__.__module__, plugin.__class__.__name__
                    ),
                    health=dict(plugin.health()),
                ).to_dict()
            )
        return descriptors

    def warmup(self) -> None:
        for scene in self.scenes():
            self._plugins[scene].warmup()

    def close(self) -> None:
        errors = []
        for scene in reversed(self.scenes()):
            try:
                self._plugins[scene].close()
            except Exception as exc:  # noqa: BLE001
                errors.append("{}: {}".format(scene, exc))
        if errors:
            raise RuntimeError("plugin close failed: {}".format("; ".join(errors)))


def load_registry_config(config_path: Path, project_root: Path) -> SceneRegistry:
    try:
        with config_path.open("r", encoding="utf-8") as file_obj:
            config = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginLoadError("cannot read plugin config {}: {}".format(config_path, exc)) from exc
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise PluginLoadError("plugin config schema_version must be 1")
    definitions = config.get("plugins")
    if not isinstance(definitions, list):
        raise PluginLoadError("plugin config plugins must be a list")
    registry = SceneRegistry()
    for index, definition in enumerate(definitions):
        if not isinstance(definition, dict):
            raise PluginLoadError("plugins[{}] must be an object".format(index))
        if not bool(definition.get("enabled", True)):
            continue
        spec = definition.get("spec")
        options = definition.get("options", {})
        if not isinstance(options, dict):
            raise PluginLoadError("plugins[{}].options must be an object".format(index))
        plugin = load_plugin_spec(str(spec), options, project_root)
        registry.register(plugin, str(spec))

    entry_point_group = str(config.get("entry_point_group", DEFAULT_ENTRY_POINT_GROUP))
    if bool(config.get("discover_entry_points", False)):
        for entry_point in _entry_points(entry_point_group):
            source = "entry_point:{}:{}".format(entry_point_group, entry_point.name)
            try:
                plugin = _instantiate(entry_point.load(), {}, source)
            except Exception as exc:  # noqa: BLE001
                raise PluginLoadError("failed to load {}: {}".format(source, exc)) from exc
            registry.register(plugin, source)
    if not registry.scenes():
        raise PluginLoadError("plugin config did not enable any scene")
    return registry


def build_default_registry(
    project_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> SceneRegistry:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    path = config_path or root / "deployment" / "framework" / "scene_plugins.json"
    if path.is_file():
        return load_registry_config(path.resolve(), root)
    raise PluginLoadError(
        "plugin config does not exist: {}; provide --plugin_config with at least one "
        "enabled scene plugin".format(path)
    )
