"""并行加载工具：线程池 + 异常聚合。"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger("hybridbrain")


class ParallelLoader:
    def __init__(self, max_workers=2, timeout=180):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.timeout = timeout
        self._errors = []

    def submit(self, name, fn, *args, **kwargs):
        def _wrap():
            t = time.time()
            try:
                r = fn(*args, **kwargs)
                print(f"[parallel] {name} done ({time.time()-t:.1f}s)",
                      flush=True)
                return r
            except Exception as e:
                print(f"[parallel] {name} FAIL: {e}", flush=True)
                raise
        return self.executor.submit(_wrap)

    def wait(self, futures):
        t0 = time.time()
        for f in as_completed(futures, timeout=self.timeout):
            try:
                f.result()
            except Exception as e:
                self._errors.append(e)
        if self._errors:
            raise self._errors[0]
        return time.time() - t0

    def shutdown(self):
        self.executor.shutdown(wait=False)
