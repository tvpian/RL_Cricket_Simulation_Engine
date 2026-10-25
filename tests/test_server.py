import json
import threading
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from cricket_rl.server import Session, handler_for


@pytest.fixture
def live_server():
    server = HTTPServer(("127.0.0.1", 0), handler_for(Session()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def request(base, path, data=None, **headers):
    payload = json.dumps(data).encode() if data is not None else None
    with urlopen(
        Request(base + path, data=payload, headers=headers), timeout=10
    ) as response:
        return json.loads(response.read())


def test_api_reset_bowl_field_and_bad_requests(live_server):
    base = live_server
    state = request(base, "/api/reset", {"seed": 12, "overs": 1, "pitch": "green"})
    assert state["score"]["runs"] == 0
    new_field = state["fielders"]
    new_field[0] = [5, 10]
    state = request(base, "/api/field", {"fielders": new_field})
    assert state["fielders"][0] == [5, 10]
    result = request(base, "/api/bowl", {"action": 4})
    assert result["score"]["attempts"] == 1
    assert len(result["delivery"]["trajectory"]) > 1
    assert len(result["delivery"]["trajectory"]) == len(
        result["delivery"]["fielder_trajectory"]
    )
    for path, data in [
        ("/api/bowl", {"action": 99}),
        ("/api/reset", {"overs": 0}),
        ("/api/field", {"fielders": [[100, 100]] * 9}),
    ]:
        with pytest.raises(HTTPError) as error:
            request(base, path, data)
        assert error.value.code == 400
    assert request(base, "/api/state")["score"]["attempts"] == 1
    with pytest.raises(HTTPError) as error:
        request(base, "/api/reset", {}, Origin="https://example.com")
    assert error.value.code == 403


def test_complete_innings_rejects_further_delivery(live_server):
    request(live_server, "/api/reset", {"overs": 1, "seed": 2})
    for _ in range(24):
        result = request(live_server, "/api/bowl", {"policy": "heuristic"})
        if result["done"]:
            break
    assert result["done"]
    assert result["score"]["balls"] == 6 or result["score"]["wickets"] == 10
    with pytest.raises(HTTPError) as error:
        request(live_server, "/api/bowl", {"action": 4})
    assert error.value.code == 400


def test_isolated_gym_api_and_remote_adapter(live_server):
    from cricket_rl.remote import RemoteEnv
    from gymnasium.utils.env_checker import check_env

    a = RemoteEnv(live_server, role="batting", balls=6)
    b = RemoteEnv(live_server, role="batting", balls=6)
    import numpy as np

    x, _ = a.reset(seed=21)
    y, _ = b.reset(seed=21)
    np.testing.assert_array_equal(x, y)
    ax = a.step(22)
    bx = b.step(22)
    np.testing.assert_array_equal(ax[0], bx[0])
    assert ax[1:] == bx[1:]
    check_env(a, skip_render_check=True)
    a.close()
    a.close()
    b.close()
    schemas = request(live_server, "/api/schema")
    assert len(schemas["environments"]) == 2


def test_dashboard_batting_role(live_server):
    state = request(live_server, "/api/reset", {"role": "batting", "seed": 3})
    assert state["role"] == "batting" and state["pending_delivery"] is not None
    step = request(live_server, "/api/bowl", {"action": 22})
    assert step["history"][0]["batting_action"] == 22
    assert step["score"]["attempts"] == 1


def test_learning_assets_are_served_with_correct_types(live_server):
    for path, mime in [
        ("/dashboard.css", "text/css"),
        ("/dashboard.js", "application/javascript"),
        ("/learn", "text/html"),
    ]:
        with urlopen(live_server + path, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith(mime)
            assert len(response.read()) > 100
    with pytest.raises(HTTPError):
        urlopen(live_server + "/../server.py", timeout=5)
    state = request(live_server, "/api/state")
    assert state["settings"]["seed"] == 42 and state["wicket_value"] == 8
