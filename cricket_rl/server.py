"""Local interactive simulator. Run: python -m cricket_rl.server."""
import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from cricket_rl import CricketEnv, BattingEnv
from cricket_rl.api import EnvironmentAPI
from cricket_rl.roles import batting_baseline
from cricket_rl.course_experiment import load_policy
from cricket_rl.experiment import baseline, load_model
import numpy as np


from cricket_rl.session import Session


def handler_for(session):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, content, mime='application/json'):
            data = json.dumps(content).encode() if mime == 'application/json' else content
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                self.respond(200, Path(__file__).with_name('dashboard.html').read_bytes(), 'text/html; charset=utf-8')
            elif path in ('/dashboard.css','/dashboard.js','/learn','/learn.html'):
                name = {'/dashboard.css':'dashboard.css','/dashboard.js':'dashboard.js','/learn':'learn.html','/learn.html':'learn.html'}[path]
                mime = {'/dashboard.css':'text/css; charset=utf-8','/dashboard.js':'application/javascript; charset=utf-8','/learn':'text/html; charset=utf-8','/learn.html':'text/html; charset=utf-8'}[path]
                self.respond(200,Path(__file__).with_name(name).read_bytes(),mime)
            elif path == '/api/schema':
                self.respond(200, {'environments': [session.environments.schema(r) for r in ('batting', 'bowling')]})
            elif path == '/api/state':
                self.respond(200, session.state())
            else:
                self.respond(404, {'error': 'Not found'})

        def do_POST(self):
            try:
                # No cross-origin mutations from arbitrary browser sites.
                origin = self.headers.get('Origin')
                if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                    self.respond(403, {'error': 'Cross-origin request rejected'})
                    return
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    raise ValueError('Expected a JSON request under 64 KiB')
                config = json.loads(self.rfile.read(length))
                if not isinstance(config, dict):
                    raise ValueError('Expected a JSON object')
                path = urlsplit(self.path).path
                if path == '/api/envs':
                    result = session.environments.create(config)
                elif path.startswith('/api/envs/'):
                    parts = path.strip('/').split('/')
                    if len(parts) != 4:
                        raise ValueError('Expected /api/envs/{id}/{operation}')
                    result = session.environments.command(parts[2], parts[3], config)
                elif path == '/api/reset':
                    result = session.reset(config)
                elif path == '/api/bowl':
                    result = session.bowl(config)
                elif path == '/api/field':
                    session.env.set_field(config.get('fielders'))
                    result = session.state()
                else:
                    self.respond(404, {'error': 'Not found'})
                    return
                self.respond(200, result)
            except (ValueError, TypeError, RuntimeError, KeyError) as error:
                self.respond(400, {'error': str(error)})

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--batting-model', type=Path)
    parser.add_argument('--bowling-model', type=Path)
    args = parser.parse_args()
    model = None
    if args.model and args.bowling_model:
        parser.error('Choose --model or --bowling-model')
    if args.bowling_model:
        model = load_policy(args.bowling_model, 'bowling')
    if args.model:
        try:
            model = load_model(args.model)
        except ValueError as error:
            parser.error(str(error))
    batting_model = load_policy(args.batting_model, 'batting') if args.batting_model else None
    server = HTTPServer((args.host, args.port), handler_for(Session(model, batting_model)))
    print(f'Cricket Strategy Lab listening on {args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
