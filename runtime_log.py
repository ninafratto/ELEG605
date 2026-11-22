#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow module: wall-clock timing of conductor stages.

    timer = RuntimeLog()
    with timer.stage("fft features"):
        ...
    timer.save(out_dir / "runtime.csv")
"""

import time
from contextlib import contextmanager

import pandas as pd


class RuntimeLog:
    def __init__(self):
        self.rows = []
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - t0
            self.rows.append({"stage": name, "seconds": seconds})
            print(f"[runtime] {name}: {seconds:.2f} s")

    def save(self, path):
        total = time.perf_counter() - self._start
        df = pd.DataFrame(self.rows + [{"stage": "TOTAL", "seconds": total}])
        df.to_csv(path, index=False)
        print(f"\n=== Runtime (s) ===\n{df.to_string(index=False, float_format=lambda x: f'{x:.2f}')}")
        print(f"saved '{path}'")
        return df
