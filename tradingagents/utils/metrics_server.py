#!/usr/bin/env python3
from __future__ import annotations
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any

try:
    from .metrics import metrics
except Exception:
    metrics = None

class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/metrics':
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; version=0.0.4')
        self.end_headers()
        content = self._render_metrics()
        self.wfile.write(content.encode('utf-8'))

    def log_message(self, format, *args):
        # Suppress default logging
        return

    def _render_metrics(self) -> str:
        if metrics is None:
            return "# metrics not available\n"
        snap = metrics.snapshot()
        lines = ["# HELP ta_counter_total Simple counters", "# TYPE ta_counter_total counter"]
        # counters
        for key, labels_to_val in (snap.get('counters') or {}).items():
            for labels, val in labels_to_val.items():
                lbl = ''
                if labels:
                    parts = [f'{k}="{v}"' for k, v in labels.items()]
                    lbl = '{' + ','.join(parts) + '}'
                lines.append(f"ta_counter_total{{metric=\"{key}\"}}{lbl} {val}")
        # histograms (we export p50/p95)
        lines.append("# HELP ta_latency_seconds Latency p50/p95/min/max")
        lines.append("# TYPE ta_latency_seconds gauge")
        for key, stats in (snap.get('hists') or {}).items():
            for p in ['p50','p95','min','max']:
                v = stats.get(p)
                if v is None:
                    continue
                lines.append(f"ta_latency_seconds{{metric=\"{key}\",quantile=\"{p}\"}} {v}")
        return "\n".join(lines) + "\n"


def start_metrics_server(host: str = '0.0.0.0', port: int = 9100):
    server = HTTPServer((host, port), _MetricsHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server