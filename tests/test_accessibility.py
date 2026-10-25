"""Automated WCAG checks complement, rather than replace, a human review."""

import os
import shutil
import threading
from pathlib import Path
from http.server import HTTPServer
import pytest

if os.environ.get("CRICKET_BROWSER_TEST") != "1":
    pytest.skip("Opt-in browser accessibility checks", allow_module_level=True)
axe = Path("node_modules/axe-core/axe.min.js")
if not axe.exists():
    pytest.skip(
        "Run npm ci to install pinned accessibility tooling", allow_module_level=True
    )
from playwright.sync_api import sync_playwright
from cricket_rl.server import handler_for, Session


@pytest.mark.parametrize("width", [390, 1440])
@pytest.mark.parametrize("mode", ["bowling", "batting", "guide"])
def test_wcag_accessible_names_contrast_and_keyboard_regions(width, mode):
    server = HTTPServer(("127.0.0.1", 0), handler_for(Session()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=shutil.which("chromium"),
                args=["--no-sandbox"],
            )
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.goto(
                f"http://127.0.0.1:{server.server_port}"
                + ("/learn" if mode == "guide" else "/")
            )
            if mode != "guide":
                page.wait_for_function("state !== null && !busy")
                if mode == "batting":
                    page.select_option("#role", "batting")
                    page.click("#reset")
                    page.wait_for_function('state.role === "batting" && !busy')
            page.add_script_tag(path=str(axe))
            violations = page.evaluate(
                'async()=> (await axe.run(document,{runOnly:{type:"tag",values:["wcag2a","wcag2aa","wcag21aa"]}})).violations'
            )
            assert not violations, [
                (v["id"], [n["target"] for n in v["nodes"]]) for v in violations
            ]
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
