from langgraph.graph import END, START, StateGraph

from app.graph.state import OrchestratorState
from app.nodes.classify import classify_node
from app.nodes.retrieve import retrieve_node
from app.nodes.vision import vision_node
from app.nodes.tools import tools_node
from app.nodes.coder import coder_node
from app.nodes.synthesize import synthesize_node
from app.nodes.finish import finish_node


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node('classify', classify_node)
    graph.add_node('retrieve', retrieve_node)
    graph.add_node('vision', vision_node)
    graph.add_node('tools', tools_node)
    graph.add_node('coder', coder_node)
    graph.add_node('synthesize', synthesize_node)
    graph.add_node('finish', finish_node)

    graph.add_edge(START, 'classify')
    graph.add_edge('classify', 'retrieve')
    graph.add_edge('retrieve', 'vision')
    graph.add_edge('vision', 'tools')
    graph.add_edge('tools', 'coder')
    graph.add_edge('coder', 'synthesize')
    graph.add_edge('synthesize', 'finish')
    graph.add_edge('finish', END)
    return graph.compile()
