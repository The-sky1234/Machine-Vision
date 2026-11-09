import io

from fastapi.testclient import TestClient
from PIL import Image

from fabqc.webapp import app


def test_home_and_scenarios():
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "产线质检台" in home.text
        sc = client.get("/api/scenarios")
        assert sc.status_code == 200
        ids = [s["id"] for s in sc.json()["scenarios"]]
        assert "coolant_drift" in ids


def test_run_coolant_and_approve():
    with TestClient(app) as client:
        res = client.post("/api/run", json={"scenario": "coolant_drift", "seed": 3})
        assert res.status_code == 200
        data = res.json()
        assert data["disclaimer"]
        flagged = [w for w in data["wafers"] if w["flagged"]]
        assert flagged, "camera should catch painted scratches"
        action = data["action"]
        if action["status"] == "pending":
            dec = client.post(
                f"/api/runs/{data['run_id']}/actions/{action['action_id']}",
                json={"approved": True},
            )
            assert dec.status_code == 200
            assert dec.json()["status"] == "approved"


def test_real_yolo_endpoint_accepts_image():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), "gray").save(buf, format="PNG")
    with TestClient(app) as client:
        status = client.get("/api/yolo/status")
        assert status.status_code == 200
        assert status.json()["available"] is True
        result = client.post(
            "/api/yolo/detect",
            files={"image": ("steel.png", buf.getvalue(), "image/png")},
        )
        assert result.status_code == 200
        body = result.json()
        assert body["model"].startswith("YOLO11n")
        assert body["annotated_image"].startswith("data:image/png;base64,")
