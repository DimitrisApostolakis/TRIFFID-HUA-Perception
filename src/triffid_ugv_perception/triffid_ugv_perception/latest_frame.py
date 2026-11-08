"""A single pending synchronized frame; inference never blocks subscription callbacks."""
import threading
import time
import math

class LatestFrameWorker:
    def __init__(self,process,period_s=1.,max_age_s=1.,on_error=None):
        if not math.isfinite(period_s) or not math.isfinite(max_age_s) or period_s<0 or max_age_s<=0:
            raise ValueError('Invalid live scheduling limits')
        self.process=process;self.period=period_s;self.max_age=max_age_s;self.on_error=on_error
        self.condition=threading.Condition();self.pending=None;self.stopped=False
        self.received=self.replaced=self.stale=self.processed=self.failed=0
        self.thread=threading.Thread(target=self._run,name='ugv-latest-frame',daemon=True)

    def start(self):self.thread.start()

    def submit(self,item):
        with self.condition:
            if self.stopped:return False
            self.received+=1
            if self.pending is not None:self.replaced+=1
            self.pending=(item,time.monotonic())
            self.condition.notify_all()
            return True

    def stats(self):
        with self.condition:
            return dict(received=self.received,replaced=self.replaced,stale=self.stale,
                processed=self.processed,failed=self.failed,pending=int(self.pending is not None))

    def _run(self):
        next_start=0.
        while True:
            with self.condition:
                while not self.stopped and (self.pending is None or time.monotonic()<next_start):
                    delay=None if self.pending is None else max(0.,next_start-time.monotonic())
                    self.condition.wait(delay)
                if self.stopped:return
                item,received=self.pending;self.pending=None
                if time.monotonic()-received>self.max_age:
                    self.stale+=1;continue
            next_start=time.monotonic()+self.period
            try:
                self.process(item,received)
                with self.condition:self.processed+=1
            except Exception as error:
                with self.condition:self.failed+=1
                if self.on_error:self.on_error(error)

    def close(self):
        with self.condition:
            self.stopped=True;self.pending=None;self.condition.notify_all()
        if self.thread.is_alive():self.thread.join(timeout=10.)
