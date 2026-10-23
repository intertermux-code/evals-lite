import sqlite3
import json
import os
import time

PRICES = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "mock": (0.0, 0.0),
}

def cost_for(model, pt, ct):
    if model not in PRICES:
        return None
    p, c = PRICES[model]
    return (pt / 1000.0) * p + (ct / 1000.0) * c

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ref TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  provider TEXT,
  model TEXT,
  total INTEGER,
  passed INTEGER,
  failed INTEGER,
  wall_time REAL,
  cost_usd REAL
);
CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  case_id TEXT NOT NULL,
  output TEXT,
  passed INTEGER NOT NULL,
  scorer_results TEXT,
  latency_ms REAL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cost_usd REAL,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""

class Ledger:
    def __init__(self, path):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def create_run(self, ref, provider, model, total, passed, failed, wall_time, cost_usd):
        cur = self.conn.execute(
            "INSERT INTO runs (ref,timestamp,provider,model,total,passed,failed,wall_time,cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
            (ref, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), provider, model, total, passed, failed, wall_time, cost_usd),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_result(self, run_id, case_id, output, passed, scorer_results, latency_ms, pt, ct, cost_usd):
        self.conn.execute(
            "INSERT INTO results (run_id,case_id,output,passed,scorer_results,latency_ms,prompt_tokens,completion_tokens,cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, case_id, output, 1 if passed else 0, json.dumps(scorer_results), latency_ms, pt, ct, cost_usd),
        )
        self.conn.commit()

    def runs(self):
        rows = self.conn.execute("SELECT id,ref,timestamp,provider,model,total,passed,failed,wall_time,cost_usd FROM runs ORDER BY id").fetchall()
        return rows

    def run_by_ref(self, ref):
        row = self.conn.execute("SELECT id,ref,timestamp,provider,model,total,passed,failed,wall_time,cost_usd FROM runs WHERE ref=? ORDER BY id DESC LIMIT 1", (ref,)).fetchone()
        return row

    def results_for(self, run_id):
        rows = self.conn.execute("SELECT case_id,output,passed,scorer_results,latency_ms,prompt_tokens,completion_tokens,cost_usd FROM results WHERE run_id=?", (run_id,)).fetchall()
        return rows

    def cost_by_model(self):
        return self.conn.execute(
            """SELECT r.model,
                      COUNT(DISTINCT r.id),
                      COUNT(s.id),
                      COALESCE(SUM(s.prompt_tokens), 0),
                      COALESCE(SUM(s.completion_tokens), 0),
                      SUM(s.cost_usd)
               FROM runs r LEFT JOIN results s ON s.run_id = r.id
               GROUP BY r.model
               ORDER BY COALESCE(SUM(s.cost_usd), 0) DESC, r.model""",
        ).fetchall()

    def cost_by_run(self):
        return self.conn.execute(
            """SELECT r.ref,
                      r.timestamp,
                      r.model,
                      r.total,
                      r.passed,
                      COALESCE(SUM(s.prompt_tokens), 0),
                      COALESCE(SUM(s.completion_tokens), 0),
                      SUM(s.cost_usd)
               FROM runs r LEFT JOIN results s ON s.run_id = r.id
               GROUP BY r.id
               ORDER BY r.id""",
        ).fetchall()
