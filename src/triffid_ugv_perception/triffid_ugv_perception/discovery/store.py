"""Bounded SQLite registry and transactional outbox; no per-frame history.

Rollback journaling avoids an unbounded WAL. Two thirds of the configured file budget are
reserved for rollback-journal records, headers and page-alignment overhead.
Update mode retains one latest and one acknowledged Feature per object.
"""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4

from .features import encode, location_ready, meaningful_change


class CapacityExceeded(RuntimeError):
    pass


class DiscoveryStore:
    def __init__(self, path, settings):
        self.path, self.settings = Path(path), settings
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._closed = False
        self._owner = self.path.with_suffix(self.path.suffix+'.lock').open('a+b')
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise RuntimeError('Another discovery service owns this database') from None
        try:
            self.db = sqlite3.connect(str(self.path), timeout=.1, check_same_thread=False, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            self.db.execute('PRAGMA foreign_keys=ON')
            self.db.execute('PRAGMA journal_mode=TRUNCATE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA cache_size=-1024')
            self.db.execute('PRAGMA auto_vacuum=INCREMENTAL')
            page_size = self.db.execute('PRAGMA page_size').fetchone()[0]
            max_pages = (settings.max_database_bytes-2*page_size)//(3*page_size+8)
            if self.db.execute('PRAGMA page_count').fetchone()[0] > max_pages:
                raise CapacityExceeded('Existing database exceeds the configured safe storage budget')
            self.db.execute(f'PRAGMA max_page_count={max_pages}')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS objects (
                    object_id TEXT PRIMARY KEY, identity_class TEXT NOT NULL,
                    class_name TEXT NOT NULL, class_id INTEGER, confidence REAL NOT NULL,
                    first_seen_ns INTEGER NOT NULL, last_seen_ns INTEGER NOT NULL,
                    bbox TEXT, appearance BLOB, geometry BLOB,
                    feature BLOB, ready INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'waiting' CHECK(status IN ('waiting','queued','sent')),
                    remote_id TEXT);
                CREATE INDEX IF NOT EXISTS objects_ready ON objects(status,ready);
                CREATE TABLE IF NOT EXISTS outbox (
                    object_id TEXT PRIMARY KEY REFERENCES objects(object_id), payload BLOB NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','inflight','blocked')),
                    attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
                    lease_until REAL NOT NULL DEFAULT 0, lease_token TEXT, last_error TEXT);
                CREATE INDEX IF NOT EXISTS outbox_due ON outbox(state,next_attempt,lease_until);
            ''')
            if 'label_state' not in {r[1] for r in self.db.execute('PRAGMA table_info(objects)')}:
                self.db.execute('ALTER TABLE objects ADD COLUMN label_state BLOB')
            # Additive migration: existing create events and remote IDs are retained.
            for table, additions in {
                'objects': [('ack_payload', 'BLOB'), ('last_delivery', 'REAL NOT NULL DEFAULT 0')],
                'outbox': [('operation', "TEXT NOT NULL DEFAULT 'create'"), ('operation_id', 'TEXT')],
            }.items():
                columns = {r[1] for r in self.db.execute(f'PRAGMA table_info({table})')}
                for name, declaration in additions:
                    if name not in columns:
                        self.db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {declaration}')
            with self.transaction():
                expected = dict(schema='triffid.discovery.sqlite.v2', namespace=settings.namespace,
                                embedding_space=settings.embedding_space,identity_policy='class_correction_v1' if settings.class_correction else 'class_aware_v1')
                for key, value in expected.items():
                    saved = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
                    if key=='schema' and saved and saved[0]=='triffid.discovery.sqlite.v1':
                        self.db.execute('UPDATE metadata SET value=? WHERE key=?', (value,key))
                        saved = (value,)
                    if key=='identity_policy' and saved is None and settings.class_correction and self.db.execute('SELECT COUNT(*) FROM objects').fetchone()[0]:
                        raise ValueError('Legacy class-aware objects require a separate class-correction database')
                    if saved and saved[0] != value:
                        raise ValueError(f'Discovery database {key} differs; use the correct mission/embedding database')
                    self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (key,value))
        except Exception:
            if hasattr(self, 'db'):
                self.db.close()
            self._owner.close()
            raise

    @contextmanager
    def transaction(self):
        with self._mutex:
            try:
                self.db.execute('BEGIN IMMEDIATE')
                yield
                self.db.execute('COMMIT')
            except Exception as error:
                if self.db.in_transaction:
                    self.db.execute('ROLLBACK')
                if isinstance(error, sqlite3.Error) and ('full' in str(error).lower() or 'locked' in str(error).lower()):
                    raise CapacityExceeded(str(error)) from error
                raise

    @staticmethod
    def _record(row):
        if row is None:
            return None
        value = dict(row)
        value['bbox'] = json.loads(value['bbox']) if value['bbox'] else None
        value['geometry'] = json.loads(value['geometry']) if value['geometry'] else None
        value['label_state'] = json.loads(value['label_state']) if value.get('label_state') else None
        return value

    def records(self):
        with self._mutex:
            return [self._record(r) for r in self.db.execute('SELECT * FROM objects')]

    def _feature(self, feature):
        if not feature:
            return None, None, False
        payload = encode(feature)
        if len(payload) > self.settings.max_payload_bytes:
            raise CapacityExceeded('Feature exceeds the per-object payload limit')
        ready = location_ready(feature)
        geometry = encode(feature['geometry']) if ready else None
        return payload, geometry, ready

    def _queue(self, object_id, payload):
        row = self.db.execute('SELECT status FROM objects WHERE object_id=?', (object_id,)).fetchone()
        if row is None or row[0] != 'waiting':
            return False
        count, size = self.db.execute('SELECT COUNT(*),COALESCE(SUM(length(payload)),0) FROM outbox').fetchone()
        if count >= self.settings.max_pending or size+len(payload) > self.settings.max_pending_bytes:
            return False
        self.db.execute('INSERT INTO outbox(object_id,payload) VALUES (?,?)', (object_id,payload))
        self.db.execute("UPDATE objects SET status='queued',feature=CASE WHEN ? THEN feature ELSE NULL END WHERE object_id=?", (self.settings.updates_enabled,object_id))
        return True

    def _validate_record(self, record):
        for key in ('object_id','identity_class','class_name'):
            value = record[key]
            if not isinstance(value,str) or not value or len(value)>128:
                raise ValueError('Invalid or oversized identity field')
        if record.get('label_state') and len(encode(record['label_state']))>self.settings.max_payload_bytes:
            raise CapacityExceeded('Class evidence exceeds its bounded per-object limit')
        appearance = record.get('appearance')
        if appearance is not None and (not isinstance(appearance,bytes) or
                len(appearance)>4*self.settings.max_appearance_dims or len(appearance)%4):
            raise ValueError('Invalid or oversized stored appearance')
        if not math.isfinite(record['confidence']) or not 0 <= record['confidence'] <= 1:
            raise ValueError('Invalid stored confidence')
        if len(encode(record.get('bbox'))) > 256:
            raise ValueError('Invalid stored image box')

    def register(self, record, feature=None):
        self._validate_record(record)
        payload, geometry, ready = self._feature(feature)
        with self.transaction():
            if self.db.execute('SELECT COUNT(*) FROM objects').fetchone()[0] >= self.settings.max_objects:
                raise CapacityExceeded('Object registry capacity reached; sent IDs are retained')
            self.db.execute('''INSERT INTO objects(object_id,identity_class,class_name,class_id,confidence,
                first_seen_ns,last_seen_ns,bbox,appearance,geometry,feature,ready,label_state)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (record['object_id'],record['identity_class'],record['class_name'],record.get('class_id'),
                 record['confidence'],record['first_seen_ns'],record['last_seen_ns'],
                 json.dumps(record.get('bbox')) if record.get('bbox') is not None else None,
                 record.get('appearance'),geometry,payload,int(ready),encode(record['label_state']) if record.get('label_state') else None))
            if ready:
                self._queue(record['object_id'],payload)

    def update_records(self, records):
        """Never overwrite queue/sent state with stale in-memory tracking state."""
        if not records:
            return
        with self.transaction():
            for record, feature in records:
                self._validate_record(record)
                payload, geometry, ready = self._feature(feature)
                self.db.execute('''UPDATE objects SET class_name=?,class_id=?,confidence=?,last_seen_ns=?,
                    bbox=?,appearance=?,label_state=?,geometry=COALESCE(?,geometry),
                    feature=CASE WHEN (status='waiting' OR ?) AND ? THEN ? ELSE feature END,
                    ready=CASE WHEN status='waiting' AND ? THEN 1 ELSE ready END
                    WHERE object_id=?''',
                    (record['class_name'],record.get('class_id'),record['confidence'],record['last_seen_ns'],
                     json.dumps(record.get('bbox')) if record.get('bbox') is not None else None,
                     record.get('appearance'),encode(record['label_state']) if record.get('label_state') else None,geometry,self.settings.updates_enabled,int(ready),payload,int(ready),record['object_id']))
                if ready:
                    self._queue(record['object_id'],payload)

    def promote_ready(self, limit=16):
        with self.transaction():
            rows = self.db.execute("SELECT object_id,feature FROM objects WHERE status='waiting' AND ready=1 AND feature IS NOT NULL LIMIT ?", (limit,)).fetchall()
            for row in rows:
                if not self._queue(row[0],row[1]):
                    break

    def _promote_updates(self, now):
        if not self.settings.updates_enabled:
            return
        rows = self.db.execute("SELECT * FROM objects WHERE status='sent' AND feature IS NOT NULL AND remote_id IS NOT NULL").fetchall()
        count, size = self.db.execute('SELECT COUNT(*),COALESCE(SUM(length(payload)),0) FROM outbox').fetchone()
        for row in rows:
            old = json.loads(row['ack_payload']) if row['ack_payload'] else None
            changed = meaningful_change(old, json.loads(row['feature']), self.settings)
            queued = self.db.execute('SELECT * FROM outbox WHERE object_id=?', (row['object_id'],)).fetchone()
            if queued:
                # Once attempted, keep exactly the same request until acknowledged.
                if queued['operation']!='update' or queued['state']!='pending' or queued['attempts']:
                    continue
                if not changed:
                    self.db.execute('DELETE FROM outbox WHERE object_id=?', (row['object_id'],))
                    count -= 1
                    size -= len(queued['payload'])
                elif queued['payload'] != row['feature'] and size-len(queued['payload'])+len(row['feature']) <= self.settings.max_pending_bytes:
                    self.db.execute('UPDATE outbox SET payload=?,operation_id=? WHERE object_id=?',
                                    (row['feature'],uuid4().hex,row['object_id']))
                    size += len(row['feature'])-len(queued['payload'])
                continue
            if not changed or count >= self.settings.max_pending or size+len(row['feature']) > self.settings.max_pending_bytes:
                continue
            self.db.execute("""INSERT INTO outbox(object_id,payload,operation,operation_id,next_attempt)
                VALUES (?,?,'update',?,?)""", (row['object_id'],row['feature'],uuid4().hex,
                    max(now,row['last_delivery']+self.settings.update_interval_s)))
            count += 1
            size += len(row['feature'])

    def claim(self, now):
        with self.transaction():
            self._promote_updates(now)
            row = self.db.execute("""SELECT outbox.*,objects.remote_id FROM outbox
                JOIN objects USING(object_id) WHERE (operation='create' OR ?)
                AND ((state='pending' AND next_attempt<=?) OR (state='inflight' AND lease_until<=?))
                ORDER BY next_attempt,object_id LIMIT 1""", (self.settings.updates_enabled,now,now)).fetchone()
            if row is None:
                return None
            token = uuid4().hex
            lease_s = max(self.settings.lease_s,self.settings.update_interval_s) if row['operation']=='update' else self.settings.lease_s
            self.db.execute("UPDATE outbox SET state='inflight',lease_until=?,lease_token=? WHERE object_id=?",
                            (now+lease_s,token,row['object_id']))
            return dict(object_id=row['object_id'],payload=bytes(row['payload']),
                        attempts=row['attempts'],lease_token=token,operation=row['operation'],
                        operation_id=row['operation_id'],remote_id=row['remote_id'])

    def acknowledge(self, object_id, token, remote_id, now=None):
        with self.transaction():
            row = self.db.execute("SELECT * FROM outbox WHERE object_id=? AND lease_token=? AND state='inflight'", (object_id,token)).fetchone()
            if row is None:
                return False
            self.db.execute("""UPDATE objects SET status='sent',remote_id=?,
                feature=CASE WHEN ? THEN feature ELSE NULL END,
                ack_payload=CASE WHEN ? THEN ? ELSE NULL END,last_delivery=? WHERE object_id=?""",
                (str(remote_id)[:256],self.settings.updates_enabled,self.settings.updates_enabled,
                 row['payload'],time.time() if now is None else now,object_id))
            self.db.execute('DELETE FROM outbox WHERE object_id=?', (object_id,))
            return True

    def fail(self, object_id, token, now, error, permanent=False):
        with self.transaction():
            row = self.db.execute("SELECT attempts,operation FROM outbox WHERE object_id=? AND lease_token=? AND state='inflight'", (object_id,token)).fetchone()
            if row is None:
                return
            attempts = min(row[0]+1,1000000)
            delay = min(self.settings.retry_max_s,self.settings.retry_initial_s*2**min(attempts-1,16))
            if row['operation']=='update':
                delay = max(delay,self.settings.update_interval_s)
            self.db.execute('''UPDATE outbox SET state=?,attempts=?,next_attempt=?,lease_until=0,
                lease_token=NULL,last_error=? WHERE object_id=?''',
                ('blocked' if permanent else 'pending',attempts,now+delay,str(error)[:256],object_id))

    def stats(self):
        with self._mutex:
            counts = dict(self.db.execute('SELECT status,COUNT(*) FROM objects GROUP BY status'))
            pending, size = self.db.execute('SELECT COUNT(*),COALESCE(SUM(length(payload)),0) FROM outbox').fetchone()
            blocked = self.db.execute("SELECT COUNT(*) FROM outbox WHERE state='blocked'").fetchone()[0]
            updates = self.db.execute("SELECT COUNT(*) FROM outbox WHERE operation='update'").fetchone()[0]
            waiting_ready = self.db.execute("SELECT COUNT(*) FROM objects WHERE status='waiting' AND ready=1").fetchone()[0]
            minimum = self.db.execute("SELECT MIN(length(feature)) FROM objects WHERE status='waiting' AND ready=1").fetchone()[0]
            queue_full = bool(waiting_ready and (pending >= self.settings.max_pending
                or size+(minimum or 0) > self.settings.max_pending_bytes))
            return dict(queue_capacity_reached=queue_full,objects=sum(counts.values()),sent=counts.get('sent',0),waiting=counts.get('waiting',0),
                        pending=pending,pending_bytes=size,blocked=blocked,waiting_ready=waiting_ready,
                        pending_updates=updates,updates_enabled=self.settings.updates_enabled)

    def ready_event_ids(self):
        """Read valid queued deliveries for diagnostics without leasing or delivery."""
        with self._mutex:
            return {row[0] for row in self.db.execute(
                "SELECT object_id FROM outbox WHERE state IN ('pending','inflight')")}

    def snapshot(self, limit=1000, max_bytes=4*1024*1024):
        features, size = [], 128
        with self._mutex:
            for row in self.db.execute('SELECT * FROM objects ORDER BY last_seen_ns DESC LIMIT ?', (limit,)):
                if row['feature']:
                    feature = json.loads(row['feature'])
                else:
                    feature = dict(type='Feature',id=row['object_id'],geometry=json.loads(row['geometry']) if row['geometry'] else None,
                        properties=dict(id=row['object_id'],object_id=row['object_id'],
                            **{'class':row['class_name']},confidence=row['confidence'],source='ugv',
                            namespace=self.settings.namespace,delivery_state=row['status']))
                encoded_size = len(encode(feature))+1
                if size+encoded_size > max_bytes:
                    break
                features.append(feature)
                size += encoded_size
        return dict(type='FeatureCollection',features=features)

    def housekeeping(self):
        with self._mutex:
            self.db.execute('PRAGMA incremental_vacuum(32)')

    def close(self):
        with self._mutex:
            if not self._closed:
                self.db.close()
                self._owner.close()
                self._closed = True
