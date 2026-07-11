from app.graph.state import OrchestratorState


def finish_node(state: OrchestratorState) -> OrchestratorState:
    if not state.get('final_answer'):
        state['final_answer'] = state.get('coder_output') or state.get('vision_notes') or state.get('retrieved_context') or 'No answer generated.'
    return state
