"""Opt-in actual-browser check: CRICKET_BROWSER_TEST=1 pytest tests/test_browser.py."""
import os
from pathlib import Path
import threading
import shutil
from http.server import HTTPServer
import pytest

if os.environ.get('CRICKET_BROWSER_TEST') != '1':
    pytest.skip('Set CRICKET_BROWSER_TEST=1 with Playwright Chromium installed', allow_module_level=True)

from playwright.sync_api import sync_playwright
from cricket_rl.server import Session, handler_for


def test_interactive_match_controls_and_replay():
    server = HTTPServer(('127.0.0.1', 0), handler_for(Session()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=shutil.which('chromium'), args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1050})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('document.getElementById("batter").textContent.length > 0')
            page.locator('#conditions summary').click()
            page.select_option('#pitch', 'dry')
            page.select_option('#overs', '1')
            page.click('#reset')
            page.wait_for_function('!document.getElementById("reset").disabled')
            page.select_option('#line', '0')
            page.select_option('#length', '2')
            page.click('#bowl')
            page.wait_for_function('document.querySelectorAll("#history tr").length === 1')
            assert page.locator('#event').inner_text() != 'New innings'
            page.click('#restart')
            page.select_option('#view', 'side')
            page.locator('#timeline').fill('0')
            page.click('#edit')
            assert page.locator('#view').input_value() == 'top'
            field = page.evaluate('state.fielders[0]')
            projected = page.evaluate('project([...state.fielders[0], 0])')
            box = page.locator('#field').bounding_box()
            x = box['x'] + projected[0] * box['width'] / 1000
            y = box['y'] + projected[1] * box['height'] / 650
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 15, y, steps=4)
            page.mouse.up()
            page.wait_for_function('!busy')
            moved = page.evaluate('state.fielders[0]')
            assert moved[0] > field[0]
            persisted = page.request.get(f'http://127.0.0.1:{server.server_port}/api/state').json()
            assert persisted['fielders'][0][0] == pytest.approx(moved[0])
            page.click('#edit')
            page.select_option('#policy', 'heuristic')
            page.click('#bowl')
            page.wait_for_function('document.querySelectorAll("#history tr").length === 2')
            with page.expect_download() as download:
                page.click('#export')
            assert download.value.suggested_filename == 'cricket-replay.json'
            page.select_option('#view', 'broadcast')
            Path('runs').mkdir(exist_ok=True)
            page.screenshot(path='runs/dashboard.png', full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.select_option('#adaptive', 'false')
            page.click('#reset')
            page.wait_for_function('!document.getElementById("reset").disabled')
            assert page.locator('#anticipation').inner_text() == 'Static batting control'
            assert page.request.get(f'http://127.0.0.1:{server.server_port}/api/state').json()['score']['adaptive_batting'] is False
            page.select_option('#role', 'batting')
            page.click('#reset')
            page.wait_for_function('!busy')
            assert page.locator('#batControls').is_visible()
            page.select_option('#policy', 'manual')
            page.select_option('#stance', '0')
            page.select_option('#direction', '4')
            page.select_option('#power', '2')
            page.click('#bowl')
            page.wait_for_function('document.querySelectorAll("#history tr").length === 1')
            remote = page.request.get(f'http://127.0.0.1:{server.server_port}/api/state').json()
            assert remote['history'][0]['batting_action'] == 14
            page.set_viewport_size({'width':1440,'height':1050})
            page.screenshot(path='runs/course-dashboard.png',full_page=True)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_offline_replay_needs_no_server(tmp_path, monkeypatch):
    import sys
    from cricket_rl.replay import main
    output = tmp_path / 'replay.html'
    monkeypatch.setattr(sys, 'argv', ['replay', '--deliveries', '3', '--output', str(output)])
    main()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=shutil.which('chromium'), args=['--no-sandbox'])
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        # Cloud Chromium blocks file URLs; render the exported bytes with all
        # networking disabled to verify the artifact is genuinely self-contained.
        page.context.set_offline(True)
        requests = []
        page.on('request', lambda request: requests.append(request.url))
        page.set_content(output.read_text())
        assert page.locator('#deliveries button').count() == 3
        page.locator('#deliveries button').nth(1).click()
        assert page.locator('#deliveries button.active').inner_text().startswith('2:')
        page.select_option('#view', 'side')
        page.locator('#timeline').fill('0')
        assert page.evaluate('frame') == 0
        assert not errors
        assert not requests
        browser.close()


def test_learning_journal_lessons_and_keyboard_field_edit():
    import json
    server = HTTPServer(('127.0.0.1', 0), handler_for(Session()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=shutil.which('chromium'), args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1050}, reduced_motion='reduce')
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('state !== null && !busy')
            page.click('#startLesson')
            page.wait_for_function('activeLesson === "first" && !busy')
            assert page.locator('#overs').input_value() == '1'
            page.select_option('#prediction', 'wicket')
            for count in range(1,4):
                page.click('#bowl')
                page.wait_for_function(f'state.history.length === {count} && !busy')
            assert 'complete' in page.locator('#lessonProgress').inner_text().lower()
            assert page.evaluate('playing') is False
            actual = page.evaluate('lastResponse')
            assert page.locator('#rewardValue').inner_text().replace('−','-').replace('+','') == str(actual['reward']).replace('.0','')
            assert page.locator('#rewardFormula').inner_text().endswith(page.locator('#rewardValue').inner_text())
            page.fill('#reflection','I need more than one lucky delivery to judge a policy.')
            with page.expect_download() as download:
                page.click('#exportSession')
            report=json.loads(Path(download.value.path()).read_text())
            assert report['version']==1 and len(report['deliveries'])==3
            assert report['reflection'].startswith('I need more')
            assert all(item['prediction']=='wicket' for item in report['deliveries'])
            page.click('#pinRun')
            assert page.locator('.snapshot').count()==1
            page.reload()
            page.wait_for_function('state !== null && !busy')
            assert page.locator('.snapshot').count()==1
            page.click('#clearRuns')
            assert page.locator('.snapshot').count()==0
            page.select_option('#lessonSelect','gaps')
            page.click('#startLesson')
            page.wait_for_function('state.role === "batting" && !busy')
            assert page.locator('#batControls').is_visible()
            assert '66' in page.locator('#observationTitle').inner_text()
            page.click('#edit')
            before=page.evaluate('state.fielders[0][0]')
            page.get_by_role('button',name='Move fielder right 3 metres').click()
            page.wait_for_function('!busy')
            assert page.evaluate('state.fielders[0][0]')==pytest.approx(before+3)
            page.fill('#seed','-1')
            page.click('#reset')
            assert 'whole-number seed' in page.locator('#error').inner_text()
            assert page.evaluate('state.role')=='batting'
            for width in (360,768,1440):
                page.set_viewport_size({'width':width,'height':1000})
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.goto(f'http://127.0.0.1:{server.server_port}/learn')
            assert page.get_by_role('heading',name='Playing, inference and training are different.').is_visible()
            page.set_viewport_size({'width':360,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            browser.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_connection_recovery_and_optional_browser_storage():
    server=HTTPServer(('127.0.0.1',0),handler_for(Session()))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path=shutil.which('chromium'),args=['--no-sandbox'])
            page=browser.new_page()
            page.add_init_script("Object.defineProperty(window, 'localStorage', {get(){throw new Error('Storage unavailable')}})")
            page.route('**/api/state',lambda route:route.fulfill(status=503,content_type='application/json',body='{"error":"Environment unavailable"}'))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function('!busy')
            assert page.locator('#bowl').is_disabled()
            assert page.locator('#reconnect').is_visible()
            assert 'Environment unavailable' in page.locator('#error').inner_text()
            page.unroute('**/api/state')
            page.click('#reconnect')
            page.wait_for_function('state !== null && !busy')
            assert page.locator('#bowl').is_enabled()
            page.click('#bowl')
            page.wait_for_function('state.history.length === 1 && !busy')
            page.click('#pinRun')
            assert page.locator('.snapshot').count()==1
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
