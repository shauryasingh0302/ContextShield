"""
Test script for verifying ContextShield POST /analyze endpoint.

Verifies:
1. HTTP 200 response when sending real image + caption.
2. risk_label is one of {"SAFE", "OFFENSIVE", "HATE", "HARASSMENT"}.
3. probabilities contains all 4 classes and sum ≈ 1.0.
4. confidence matches predicted class probability.
5. risk_score is numeric and in [0, 100].
6. detected_text is present.
7. explanation and suggestion are null/honest placeholders.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
from classifier_model import CLASS_NAMES

client = TestClient(app)

def test_analyze_endpoint():
    project_root = backend_dir.parent
    test_image_path = project_root / "dataset" / "final" / "images" / "h6Pkqkr.png"
    
    assert test_image_path.exists(), f"Test image not found at {test_image_path}"
    
    test_caption = "After the Bern subsides , get ready to ... Feel The Johnson"
    
    print(f"Testing POST /analyze with image: {test_image_path.name}")
    print(f"Caption: '{test_caption}'")
    
    with open(test_image_path, "rb") as f:
        files = {"image": (test_image_path.name, f, "image/png")}
        data = {"caption": test_caption}
        response = client.post("/analyze", files=files, data=data)
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    
    res = response.json()
    print("\n--- Response Payload ---")
    import json
    print(json.dumps(res, indent=2))
    
    # 1. Verify required keys
    required_keys = [
        "risk_label",
        "risk_score",
        "confidence",
        "probabilities",
        "detected_text",
        "explanation",
        "suggestion",
    ]
    for key in required_keys:
        assert key in res, f"Missing required response field: '{key}'"
        
    # 2. Verify risk_label is one of 4 classes
    risk_label = res["risk_label"]
    assert risk_label in CLASS_NAMES, f"risk_label '{risk_label}' not in {CLASS_NAMES}"
    
    # 3. Verify probabilities contains all 4 classes and sums to ~1.0
    probabilities = res["probabilities"]
    assert len(probabilities) == 4, f"Expected 4 classes in probabilities, got {len(probabilities)}"
    for c_name in CLASS_NAMES:
        assert c_name in probabilities, f"Missing class '{c_name}' in probabilities"
        assert 0.0 <= probabilities[c_name] <= 1.0, f"Invalid probability for {c_name}: {probabilities[c_name]}"
        
    prob_sum = sum(probabilities.values())
    assert abs(prob_sum - 1.0) < 0.02, f"Probabilities do not sum to ~1.0: {prob_sum}"
    
    # 4. Verify confidence matches predicted class probability
    confidence = res["confidence"]
    assert abs(confidence - probabilities[risk_label]) < 1e-4, (
        f"confidence ({confidence}) does not match predicted class probability ({probabilities[risk_label]})"
    )
    
    # 5. Verify risk_score is numeric and within [0, 100]
    risk_score = res["risk_score"]
    assert isinstance(risk_score, (int, float)), f"risk_score is not numeric: {type(risk_score)}"
    assert 0.0 <= risk_score <= 100.0, f"risk_score out of range [0, 100]: {risk_score}"
    
    # 6. Verify detected_text
    assert isinstance(res["detected_text"], str), f"detected_text is not a string: {type(res['detected_text'])}"
    
    # 7. Verify explanation and suggestion are null
    assert res["explanation"] is None, f"Expected explanation to be null, got: {res['explanation']}"
    assert res["suggestion"] is None, f"Expected suggestion to be null, got: {res['suggestion']}"

    print("\n[PASS] All endpoint assertions passed successfully!")


if __name__ == "__main__":
    test_analyze_endpoint()
