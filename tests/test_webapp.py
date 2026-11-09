from fastapi.testclient import TestClient

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
