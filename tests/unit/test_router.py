from app.graph.router import classify
from app.settings import Settings


def test_classify_sets_expected_flags():
    flags = classify('fix the dockerfile and check logs', [], Settings())

    assert flags['needs_coder'] is True
    assert flags['needs_tools'] is True
    assert flags['needs_vision'] is False
    assert flags['needs_retrieval'] is False


def test_classify_detects_images():
    flags = classify('', ['data:image/png;base64,abc'], Settings())

    assert flags['needs_vision'] is True
