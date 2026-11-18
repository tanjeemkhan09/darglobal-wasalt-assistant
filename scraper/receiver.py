"""Tiny local receiver: lets the in-browser crawler POST extracted pages to disk."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
OUT = sys.argv[1]
class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with open(OUT, "ab") as f:
            f.write(body.strip() + b"\n")
        self.send_response(200); self._cors(); self.end_headers(); self.wfile.write(b"ok")
    def do_GET(self):
        # Collector page: the crawler stashes results in window.name, navigates here, and we POST them to disk.
        page = b"""<html><body>collecting...<script>
fetch('/', {method:'POST', body: window.name}).then(function(){document.body.textContent='saved '+window.name.length; window.name='';});
</script></body></html>"""
        self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(page)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", 8765), H).serve_forever()
