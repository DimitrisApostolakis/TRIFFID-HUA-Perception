"""One background sender; persistent leases and retries survive crashes."""
import threading
import time

from .transport import PermanentDeliveryError


class OutboxWorker:
    def __init__(self, store, transport, logger, clock=time.time, poll_s=.25):
        self.store, self.transport, self.logger = store, transport, logger
        self.clock, self.poll_s = clock, poll_s
        if getattr(transport,'timeout_s',0) >= store.settings.lease_s:
            raise ValueError('API timeout must be shorter than the outbox lease')
        self._stop = threading.Event()
        self.thread = None

    def pump_once(self):
        self.store.promote_ready()
        event = self.store.claim(self.clock())
        if event is None:
            return False
        try:
            if event['operation']=='update':
                remote_id = self.transport.update(event['object_id'],event['remote_id'],
                    event['payload'],event['operation_id'])
            else:
                remote_id = self.transport.create(event['object_id'],event['payload'])
            if not remote_id:
                raise RuntimeError('Empty delivery acknowledgement')
        except Exception as error:
            self.store.fail(event['object_id'],event['lease_token'],self.clock(),error,
                            permanent=isinstance(error,PermanentDeliveryError))
            self.logger.warning('Object %s deferred (%s): %s',event['operation'],event['object_id'],str(error)[:256])
        else:
            # If the process fails here, the lease eventually expires and
            # the server must return the same object for this same key.
            self.store.acknowledge(event['object_id'],event['lease_token'],remote_id,self.clock())
        return True

    def _run(self):
        while not self._stop.is_set():
            try:
                worked = self.pump_once()
            except Exception as error:
                self.logger.warning('Upload worker paused: %s',str(error)[:256])
                worked = False
            self._stop.wait(.01 if worked else self.poll_s)

    def start(self):
        if self.thread is not None:
            raise RuntimeError('Worker already started')
        self.thread = threading.Thread(target=self._run,name='ugv-object-uploader',daemon=True)
        self.thread.start()

    def stop(self, timeout_s=6.):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout_s)
            return not self.thread.is_alive()
        return True
