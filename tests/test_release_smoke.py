from pathlib import Path
import json

def test_synthetic_example_is_marked():
    p=Path(__file__).parents[1]/"example_data/synthetic_structured_input.json"
    assert json.loads(p.read_text(encoding="utf-8"))["synthetic"] is True
