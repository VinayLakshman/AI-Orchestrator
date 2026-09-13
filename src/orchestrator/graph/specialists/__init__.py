"""Specialist node entry points.

Implementations remain in ``graph.nodes`` during the compatibility migration;
this package gives each capability a stable, discoverable ownership boundary.
"""

from .coding import make_coder_node
from .finalization import make_clarify_node, make_finalize_node
from .knowledge import make_knowledge_node
from .reasoning import make_reasoning_node
from .tools import make_tools_node
from .vision import make_vision_node
from .web import make_web_node

__all__ = [
    "make_clarify_node",
    "make_coder_node",
    "make_finalize_node",
    "make_knowledge_node",
    "make_reasoning_node",
    "make_tools_node",
    "make_vision_node",
    "make_web_node",
]
