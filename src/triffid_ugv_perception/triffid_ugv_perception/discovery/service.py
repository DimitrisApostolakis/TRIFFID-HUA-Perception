"""Local identity updates and optional durable background PUT/PATCH delivery."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time

from .features import identified_feature, location_ready
from .registry import PersistentRegistry
from .store import DiscoveryStore, CapacityExceeded
from .outbox import OutboxWorker
from .transport import CreateOnlyHTTPTransport, FeatureHTTPTransport


class DiscoveryService:
    def __init__(self, path, settings, api_enabled=False, api_url='',
                 uniqueness_contract=False, transport=None, clock=time.monotonic):
        # Fail before opening state or starting a thread if remote semantics
        # have not been explicitly established. Disabled means zero requests.
        if api_enabled and not uniqueness_contract:
            raise ValueError('Verify server UUID uniqueness before enabling API delivery')
        self.settings = settings
        self.store = DiscoveryStore(path,settings)
        try:
            self.registry = PersistentRegistry(self.store,settings,clock=clock)
            self.logger = logging.getLogger(f'triffid.discovery.{id(self)}')
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            logs = Path(path).parent/'logs'
            logs.mkdir(exist_ok=True)
            self.handler = RotatingFileHandler(logs/'discovery.log',maxBytes=settings.log_bytes,
                                                backupCount=settings.log_backups,encoding='utf-8')
            self.logger.addHandler(self.handler)
        except Exception:
            self.store.close()
            raise
        self.worker = None
        self._last_pressure = None
        self._closed = False
        self._last_maintenance = clock()
        self.clock = clock
        if api_enabled:
            try:
                transport_class = FeatureHTTPTransport if settings.updates_enabled else CreateOnlyHTTPTransport
                client = transport or transport_class(api_url,settings.namespace,uniqueness_contract)
                self.worker = OutboxWorker(self.store,client,self.logger)
                self.worker.start()
            except Exception:
                self.handler.close()
                self.logger.removeHandler(self.handler)
                self.store.close()
                raise

    def process(self, tracks, features, timestamp_ns):
        result = self.registry.update(tracks,features,timestamp_ns)
        output = []
        for track, feature in zip(result.observations,features):
            if not feature:
                continue
            identity = track['object_id']
            item = identified_feature(feature,identity,self.settings.namespace) if identity else dict(feature)
            item['properties'] = dict(item.get('properties',{}),object_id=identity,
                                      discovery_state=track['discovery_state'])
            # Neither local metres nor a guessed north-facing location is a
            # valid map point. Tracking continues while this location waits.
            if not location_ready(item):
                item = dict(item,geometry=None)
            output.append(item)
        if self.clock()-self._last_maintenance >= self.settings.flush_interval_s:
            try:
                self.store.promote_ready()
                self.store.housekeeping()
            except CapacityExceeded as error:
                self.registry.paused_reason = str(error)
            self._last_maintenance = self.clock()
            status = self.status()
            pressure = status['paused_reason']
            if pressure != self._last_pressure:
                if pressure:
                    self.logger.warning('New uploads paused: %s',pressure)
                else:
                    self.logger.info('New-upload capacity available')
                self._last_pressure = pressure
        return result, dict(type='FeatureCollection',features=output)

    def status(self):
        value = self.store.stats()
        value['paused_reason'] = self.registry.paused_reason
        if value['objects'] >= self.settings.max_objects:
            value['paused_reason'] = 'Object registry capacity reached; sent IDs are retained'
        if value['queue_capacity_reached']:
            value['paused_reason'] = 'Outbox capacity reached; pending events and sent identities retained'
        return value

    def close(self):
        if self._closed:
            return
        if self.worker and not self.worker.stop():
            self.logger.warning('Uploader still stopping; durable state remains open until process exit')
            return
        try:
            self.registry.flush()
        except CapacityExceeded as error:
            self.logger.warning('Latest observation refresh could not be saved: %s',str(error)[:256])
        finally:
            self.store.close()
            self.handler.close()
            self.logger.removeHandler(self.handler)
            self._closed = True
