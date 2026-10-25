"""Build site first, then CRICKET_STATIC_TEST=1 pytest tests/test_static_site.py."""

import os
import shutil
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import pytest

if os.environ.get("CRICKET_STATIC_TEST") != "1":
    pytest.skip("Opt-in bundled browser runtime test", allow_module_level=True)
from playwright.sync_api import sync_playwright


def test_bundled_browser_roles_without_external_requests(tmp_path):
    shutil.copytree(Path("site"), tmp_path / "project" / "demo")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=shutil.which("chromium"),
                args=["--no-sandbox"],
            )
            context = browser.new_context()
            external = []

            def route(request):
                if not request.request.url.startswith(origin + "/"):
                    external.append(request.request.url)
                    request.abort()
                else:
                    request.continue_()

            context.route("**/*", route)
            page = context.new_page()
            page.goto(origin + "/project/demo/")
            page.wait_for_function(
                'state !== null || document.getElementById("error").textContent.length > 0',
                timeout=60000,
            )
            assert page.evaluate("state !== null"), page.locator("#error").inner_text()
            page.click("#startLesson")
            page.wait_for_function('activeLesson === "first" && !busy')
            page.click("#bowl")
            page.wait_for_function("state.history.length === 1 && !busy")
            page.select_option("#lessonSelect", "gaps")
            page.click("#startLesson")
            page.wait_for_function('state.role === "batting" && !busy')
            page.click("#bowl")
            page.wait_for_function("state.history.length === 1 && !busy")
            assert page.evaluate("state.history[0].batting_action") >= 0
            assert page.locator("#policy option[value=ppo]").is_enabled()
            page.select_option("#policy", "ppo")
            page.click("#bowl")
            page.wait_for_function("state.history.length === 2 && !busy")
            assert page.evaluate("state.history[1].policy") == "ppo"
            assert page.evaluate("state.model_label") == "Tabular Q-learning"
            page.goto(origin + "/project/demo/learn.html")
            assert page.get_by_role(
                "heading", name="Playing, inference and training are different."
            ).is_visible()
            assert not external, external
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_browser_runtime_recovers_from_startup_and_midgame_failure(tmp_path):
    shutil.copytree(Path("site"), tmp_path / "demo")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=shutil.which("chromium"),
                args=["--no-sandbox"],
            )
            page = browser.new_page()
            page.route(
                "**/app-wheel.json",
                lambda route: route.fulfill(status=503, body="unavailable"),
            )
            page.goto(origin + "/demo/")
            page.wait_for_function(
                '!busy && document.getElementById("reconnect").hidden === false',
                timeout=60000,
            )
            assert page.locator("#bowl").is_disabled()
            page.unroute("**/app-wheel.json")
            page.click("#reconnect")
            page.wait_for_function("state !== null && !busy", timeout=60000)
            page.click("#bowl")
            page.wait_for_function("state.history.length === 1 && !busy")
            page.fill("#reflection", "Keep this note after a runtime failure.")
            page.evaluate("worker.onerror()")
            page.click("#bowl")
            page.wait_for_function(
                '!busy && document.getElementById("reconnect").hidden === false'
            )
            assert page.locator("#bowl").is_disabled()
            page.click("#reconnect")
            page.wait_for_function("state.history.length === 0 && !busy", timeout=60000)
            assert (
                page.evaluate("archivedRuns[0].reflection")
                == "Keep this note after a runtime failure."
            )
            assert page.evaluate("delivery") is None
            page.click("#bowl")
            page.wait_for_function("state.history.length === 1 && !busy")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_bad_demo_weights_do_not_break_manual_play(tmp_path):
    shutil.copytree(Path("site"), tmp_path / "demo")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=shutil.which("chromium"),
                args=["--no-sandbox"],
            )
            page = browser.new_page()
            page.route(
                "**/models/batting.npz",
                lambda route: route.fulfill(status=200, body="corrupt weights"),
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/demo/")
            page.wait_for_function(
                'state !== null || document.getElementById("error").textContent.length > 0',
                timeout=60000,
            )
            assert page.evaluate("state !== null"), page.locator("#error").inner_text()
            assert page.locator("#modelAvailability").is_visible()
            assert page.locator("#policy option[value=ppo]").is_disabled()
            page.click("#bowl")
            page.wait_for_function("state.history.length === 1 && !busy")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
