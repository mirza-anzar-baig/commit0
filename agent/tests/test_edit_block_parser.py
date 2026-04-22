"""Tests for SEARCH/REPLACE edit block parsing in openhands_formatter."""

from agent.openhands_formatter import parse_edit_blocks


def test_fenced_block_with_filename_inside():
    content = (
        "Here is the fix:\n\n"
        "```python\n"
        "pipfile/api.py\n"
        "<<<<<<< SEARCH\n"
        "def load(data):\n"
        "    pass\n"
        "=======\n"
        "def load(data):\n"
        "    return parse(data)\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    reasoning, edits = parse_edit_blocks(content)
    assert len(edits) == 1
    assert edits[0].path == "pipfile/api.py"
    assert "pass" in edits[0].old_str
    assert "parse(data)" in edits[0].new_str
    assert "Here is the fix:" in reasoning


def test_bare_search_replace():
    content = (
        "Fix:\n\n"
        "pipfile/api.py\n"
        "<<<<<<< SEARCH\n"
        "def load(data):\n"
        "    pass\n"
        "=======\n"
        "def load(data):\n"
        "    return parse(data)\n"
        ">>>>>>> REPLACE\n"
    )
    reasoning, edits = parse_edit_blocks(content)
    assert len(edits) == 1
    assert edits[0].path == "pipfile/api.py"


def test_fenced_filename_before_fence():
    content = (
        "pipfile/api.py\n"
        "```python\n"
        "<<<<<<< SEARCH\n"
        "def load(data):\n"
        "    pass\n"
        "=======\n"
        "def load(data):\n"
        "    return parse(data)\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    reasoning, edits = parse_edit_blocks(content)
    assert len(edits) == 1
    assert edits[0].path == "pipfile/api.py"


def test_multiple_edits_in_fenced_block():
    content = (
        "```python\n"
        "pipfile/api.py\n"
        "<<<<<<< SEARCH\n"
        "def load(data):\n"
        "    pass\n"
        "=======\n"
        "def load(data):\n"
        "    return parse(data)\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "pipfile/api.py\n"
        "<<<<<<< SEARCH\n"
        "def dump(data):\n"
        "    pass\n"
        "=======\n"
        "def dump(data):\n"
        "    return serialize(data)\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    reasoning, edits = parse_edit_blocks(content)
    assert len(edits) == 2
    assert edits[0].path == "pipfile/api.py"
    assert edits[1].path == "pipfile/api.py"
    assert "parse(data)" in edits[0].new_str
    assert "serialize(data)" in edits[1].new_str


def test_no_search_replace():
    content = "This is just reasoning text with no edits."
    reasoning, edits = parse_edit_blocks(content)
    assert len(edits) == 0
    assert "reasoning text" in reasoning


def test_fenced_block_no_filename_no_default():
    content = "```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n```\n"
    reasoning, edits = parse_edit_blocks(content)
    assert len(edits) == 0
