"""Schema aliases must not make Vulture's generated reference files invalid Python."""

import ast

from scripts.run_vulture import _scan_string_type_node, _write_phantom


def test_keyword_field_alias_does_not_break_string_type_references(tmp_path):
    annotation = ast.parse('Annotated["ArtifactId", Field(alias="from")]', mode="eval").body
    refs: set[str] = set()
    _scan_string_type_node(annotation, refs)
    _write_phantom(tmp_path, "refs.py", refs)
    tree = ast.parse((tmp_path / "refs.py").read_text())
    assert {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} == {"_", "ArtifactId"}
