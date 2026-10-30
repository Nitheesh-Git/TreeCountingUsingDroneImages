import sys
import time

class tqdm:
    def __init__(self, iterable=None, desc='', total=None, initial=0, bar_format=None, **kwargs):
        self.iterable = iterable
        self.desc = desc or ''
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, '__len__') else None)
        self.n = initial
        self.start_time = time.time()
        self.last_print_time = 0.0
        if iterable is None and self.desc:
            print(self.desc, flush=True)

    def __iter__(self):
        if self.iterable is None:
            return
        for idx, item in enumerate(self.iterable):
            self.n = idx + 1
            now = time.time()
            if idx == 0 or self.n == self.total or (now - self.last_print_time) >= 4.0:
                self.last_print_time = now
                elapsed = max(now - self.start_time, 1e-6)
                rate = self.n / elapsed
                if self.total:
                    pct = 100.0 * self.n / self.total
                    rem = (self.total - self.n) / max(rate, 1e-6)
                    print(f"{self.desc} [{self.n}/{self.total} ({pct:.0f}%) | {1.0/max(rate,1e-6):.2f}s/it | {elapsed:.1f}s<{rem:.1f}s]", flush=True)
                else:
                    print(f"{self.desc} [{self.n} it | {elapsed:.1f}s]", flush=True)
            yield item

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or ''

    def update(self, n=1):
        self.n += n

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
