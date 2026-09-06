"""
Comprehensive test suite for ContextShield Explanation & Safer-Caption Suggestion System.

Verifies:
1. SAFE returns suggestion None (null in JSON).
2. All classes return a non-empty explanation.
3. Suggestions are non-empty for risky classes (OFFENSIVE, HATE, HARASSMENT).
4. No input caption is modified or overwritten.
5. Rule-based truthfulness: does not claim specific words were detected unless they actually appear.
6. End-to-end POST /analyze endpoint maintains all existing risk classification metrics
   (risk_label, risk_score, confidence, probabilities) while providing explanation & suggestion.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from explanation_service import (
    generate_explanation,
    generate_suggestion,
    find_flagged_words,
    neutralize_text,
)
from main import app
from classifier_model import CLASS_NAMES

client = TestClient(app)


def test_unit_explanation_and_suggestion():
    """Unit tests for explanation_service covering all 4 risk classes."""
    classes = ["SAFE", "OFFENSIVE", "HATE", "HARASSMENT"]
    
    # 1. SAFE Post
    safe_caption = "Hope everyone has a wonderful and productive day!"
    safe_copy = str(safe_caption)
    safe_exp = generate_explanation("SAFE", 0.95, safe_caption, "Good morning world")
    safe_sug = generate_suggestion("SAFE", safe_caption, "Good morning world")
    
    assert isinstance(safe_exp, str) and len(safe_exp) > 0, "SAFE explanation must be non-empty"
    assert "no significant risk" in safe_exp.lower(), "SAFE explanation should state no significant risk detected"
    assert safe_sug is None, f"SAFE suggestion must be None, got: {safe_sug}"
    assert safe_caption == safe_copy, "Input caption must not be mutated"
    print("[PASS] Unit test: SAFE label verification")

    # 2. OFFENSIVE Post without specific flagged word
    off_caption = "This whole thing is a complete joke"
    off_copy = str(off_caption)
    off_exp = generate_explanation("OFFENSIVE", 0.82, off_caption, "")
    off_sug = generate_suggestion("OFFENSIVE", off_caption, "")
    
    assert isinstance(off_exp, str) and len(off_exp) > 0, "OFFENSIVE explanation must be non-empty"
    assert "offensive" in off_exp.lower() or "insulting" in off_exp.lower()
    assert "detected term" not in off_exp, "Must not claim words detected when none were present"
    assert isinstance(off_sug, str) and len(off_sug) > 0, "OFFENSIVE suggestion must be non-empty"
    assert off_caption == off_copy, "Input caption must not be mutated"
    print("[PASS] Unit test: OFFENSIVE (generic) verification")

    # 3. OFFENSIVE Post with flagged word (neutralization test)
    vulgar_caption = "You are an idiot and this is stupid"
    vulgar_copy = str(vulgar_caption)
    v_flagged = find_flagged_words(vulgar_caption)
    assert "idiot" in v_flagged and "stupid" in v_flagged
    v_exp = generate_explanation("OFFENSIVE", 0.88, vulgar_caption, "")
    v_sug = generate_suggestion("OFFENSIVE", vulgar_caption, "")
    
    assert "'idiot'" in v_exp and "'stupid'" in v_exp, "Explanation should mention detected words"
    assert isinstance(v_sug, str) and len(v_sug) > 0
    assert "idiot" not in v_sug.lower() and "stupid" not in v_sug.lower(), "Suggestion should neutralize detected words"
    assert vulgar_caption == vulgar_copy, "Input caption must not be mutated"
    print("[PASS] Unit test: OFFENSIVE (with neutralization) verification")

    # 4. HATE Post
    hate_caption = "They should all be eliminated from our society"
    hate_copy = str(hate_caption)
    hate_exp = generate_explanation("HATE", 0.91, hate_caption, "")
    hate_sug = generate_suggestion("HATE", hate_caption, "")
    
    assert isinstance(hate_exp, str) and len(hate_exp) > 0, "HATE explanation must be non-empty"
    assert "hate" in hate_exp.lower() or "identity" in hate_exp.lower()
    assert isinstance(hate_sug, str) and len(hate_sug) > 0, "HATE suggestion must be non-empty"
    assert hate_caption == hate_copy, "Input caption must not be mutated"
    print("[PASS] Unit test: HATE label verification")

    # 5. HARASSMENT Post
    harass_caption = "Look at this loser, let's make their life hell"
    harass_copy = str(harass_caption)
    harass_exp = generate_explanation("HARASSMENT", 0.85, harass_caption, "")
    harass_sug = generate_suggestion("HARASSMENT", harass_caption, "")
    
    assert isinstance(harass_exp, str) and len(harass_exp) > 0, "HARASSMENT explanation must be non-empty"
    assert "individual" in harass_exp.lower() or "insult" in harass_exp.lower() or "target" in harass_exp.lower()
    assert isinstance(harass_sug, str) and len(harass_sug) > 0, "HARASSMENT suggestion must be non-empty"
    assert harass_caption == harass_copy, "Input caption must not be mutated"
    print("[PASS] Unit test: HARASSMENT label verification")


def test_endpoint_with_real_image():
    """End-to-end integration test of POST /analyze with explanation and suggestion."""
    project_root = backend_dir.parent
    test_image_path = project_root / "dataset" / "final" / "images" / "h6Pkqkr.png"
    assert test_image_path.exists(), f"Image not found at {test_image_path}"
    
    submitted_caption = "After the Bern subsides , get ready to ... Feel The Johnson"
    
    with open(test_image_path, "rb") as f:
        files = {"image": (test_image_path.name, f, "image/png")}
        data = {"caption": submitted_caption}
        response = client.post("/analyze", files=files, data=data)
        
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    res = response.json()
    
    # 1. Existing metric assertions
    assert res["risk_label"] in CLASS_NAMES
    assert isinstance(res["risk_score"], (int, float)) and 0.0 <= res["risk_score"] <= 100.0
    assert isinstance(res["confidence"], float) and 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 4
    for c in CLASS_NAMES:
        assert c in res["probabilities"]
    assert abs(sum(res["probabilities"].values()) - 1.0) < 0.02
    
    # 2. Original caption preservation
    assert res["caption"] == submitted_caption.strip()
    
    # 3. Explanation assertions
    assert "explanation" in res
    assert isinstance(res["explanation"], str) and len(res["explanation"]) > 0
    
    # 4. Suggestion assertions
    assert "suggestion" in res
    if res["risk_label"] == "SAFE":
        assert res["suggestion"] is None, f"SAFE must return suggestion null, got: {res['suggestion']}"
    else:
        assert isinstance(res["suggestion"], str) and len(res["suggestion"]) > 0, "Risky post must return non-empty suggestion"

    print("\n[PASS] Integration test: POST /analyze returned valid response:")
    print(f"  Risk Label:   {res['risk_label']}")
    print(f"  Confidence:   {res['confidence'] * 100:.1f}%")
    print(f"  Explanation:  {res['explanation']}")
    print(f"  Suggestion:   {res['suggestion']}")
    print(f"  Original Cap: {res['caption']}")


if __name__ == "__main__":
    test_unit_explanation_and_suggestion()
    test_endpoint_with_real_image()
    print("\nAll explanation and suggestion tests PASSED!")
