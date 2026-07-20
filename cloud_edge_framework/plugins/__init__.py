"""用途：导出场景无关的插件基类；具体场景插件由配置动态加载。"""

from cloud_edge_framework.plugins.base import ScenePlugin

__all__ = [
    "ScenePlugin",
]
