"""
Verification script: send a real image + caption to the running backend (http://127.0.0.1:8000/analyze),
mimicking the frontend's exact fetch request, and verify all UI-bound fields.
"""

import urllib.request
import json
from pathlib import Path

img_path = Path("dataset/final/images/h6Pkqkr.png")
assert img_path.exists(), f"Image not found at {img_path}"

boundary = "----WebKitFormBoundaryContextShieldTest"

with open(img_path, "rb") as f:
    img_bytes = f.read()

caption = "After the Bern subsides , get ready to ... Feel The Johnson"

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="caption"\r\n\r\n'
    f"{caption}\r\n"
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="image"; filename="{img_path.name}"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8000/analyze",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))

print("=" * 60)
print("REAL BACKEND RESPONSE (VERIFIED END-TO-END)")
print("=" * 60)
print("1. Risk Category:      ", res.get("risk_label"))
print("2. Risk Score:         ", f"{res.get('risk_score')}%")
print("3. Confidence:         ", f"{res.get('confidence') * 100:.1f}%")
print("4. Probability Breakdown:")
for cat, prob in res.get("probabilities", {}).items():
    print(f"     - {cat:<12}: {prob * 100:.1f}%")
print("5. Detected OCR Text:  ", repr(res.get("detected_text")))
print("6. Explanation:        ", res.get("explanation"))
print("7. Caption Suggestion: ", res.get("suggestion"))
print("=" * 60)

# Verify all values are valid
assert res["risk_label"] in ["SAFE", "OFFENSIVE", "HATE", "HARASSMENT"]
assert isinstance(res["risk_score"], (int, float))
assert isinstance(res["confidence"], float)
assert len(res["probabilities"]) == 4
assert res["detected_text"] is not None
assert res["explanation"] is not None
print("All frontend-bound values successfully confirmed from live API response!")
