"""Shared image/document admission, channel updates and independent lease heartbeats."""
from .pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Empty, Full, Queue
from threading import Condition, Event, Lock
import time
from uuid import uuid4

from .protocol import ControlFailure, NodeFailure, timestamp
from .operations import NetworkLog, utc_now

LOG = logging.getLogger('classic-node')


class Page:
    def __init__(self, lease, server_time, sent_at):
        self.lease = lease
        self.condition = Condition()
        self.deadline = 0
        self.renewed_at = 0
        self.terminal = None
        self.stopped = False
        self.phase = 'queued'
        self.step = 'download'
        self.ready_at = time.monotonic()
        self.timings = {}
        self.future = self.analysis_future = self.pending_error = None
        self.analysis_accepted = False
        self.reserved = 0
        self.render_reserved = 0
        self.mask_reserved = 0
        self.masks = {}
        self.next_stop = 0
        self.data = self.metadata = self.rgb = self.cleaned = self.analysis = self.alpha = self.completion = None
        self.rendered = None
        self.received_at = sent_at
        self.translations = lease.get('translations')
        self.update(lease, server_time, sent_at)

    def update(self, reply, server_time=None, sent_at=None):
        with self.condition:
            status = reply['status']
            if status == 'terminal':
                self.terminal = reply
            elif status == 'stop':
                self.stopped = True
            elif not self.stopped and not self.terminal:
                if self.deadline and time.monotonic() >= self.deadline:
                    self.stopped = True
                    self.condition.notify_all()
                    return
                if sent_at is not None and sent_at >= self.renewed_at:
                    # Only registration/claim/heartbeat renew execution rights.
                    # A channel update carries no expiry and never extends them.
                    self.deadline = sent_at + max(0, timestamp(reply['expires_at']) - timestamp(server_time) - 2)
                    self.lease['expires_at'] = reply['expires_at']
                    self.renewed_at = sent_at
                if reply.get('translations'):
                    self.translations = reply['translations']
            self.condition.notify_all()

    def check(self):
        with self.condition:
            if self.terminal or self.stopped or time.monotonic() >= self.deadline:
                self.stopped = True
                raise NodeFailure('LEASE_STOPPED')


class Agent:
    def __init__(self, config, runtime, transport, journal, *, stop=None, operations=None):
        self.local, self.runtime, self.transport, self.journal = config, runtime, transport, journal
        self.config = None
        self.config_lock = Lock()
        self.pages = {}
        self.wake = Event()
        self.pipeline = Pipeline(self)
        self.next_claim = 0
        self.heartbeat_pool = ThreadPoolExecutor(1, thread_name_prefix='heartbeat')
        self.claim_pool = ThreadPoolExecutor(1, thread_name_prefix='claim')
        self.channel_pool = ThreadPoolExecutor(1, thread_name_prefix='channel')
        self.heartbeat_future = self.claim_future = self.channel_future = None
        self.updates = Queue(maxsize=4)
        self.channel_backoff = 1
        self.server_clock_offset = 0
        self.next_channel = 0
        self.stop = stop if stop is not None else Event()
        self.operations = operations
        self.network_log = NetworkLog()
        self.heartbeat_at = None
        self.heartbeat_monotonic = 0
        self.heartbeat_logged = 0
        self.last_heartbeat = 0
        from .document_runner import DocumentRunner
        self.documents = DocumentRunner(self)

    def active_pages(self):
        return list(self.pages.values()) + list(self.documents.pages.values())

    def connected(self, action):
        self.network_log.succeeded(action)
        self.heartbeat_at = utc_now()
        self.heartbeat_monotonic = time.monotonic()
        if action == 'registration' or self.heartbeat_monotonic - self.heartbeat_logged >= 60:
            LOG.info('event=%s config_version=%s active_leases=%s', action, self.config['version'], len(self.active_pages()))
            self.heartbeat_logged = self.heartbeat_monotonic

    def apply_config(self, config):
        with self.config_lock:
            if config is not None and (self.config is None or config['version'] >= self.config['version']):
                changed = self.config is None or config['version'] != self.config['version']
                self.config = config
                self.transport.control.timeout = config['request_seconds']
                if changed:
                    self.next_claim = 0
                    self.wake.set()

    def claim_capacity(self):
        if not self.config['enabled'] or self.stop.is_set():
            return 0
        slots = min(self.config['execution_slots'], self.local['max_leases']) - len(self.active_pages())
        return max(0, min(slots, self.pipeline.claim_capacity()))

    def request_claim(self):
        with self.config_lock:
            self.next_claim = 0
        self.wake.set()

    def lease_entries(self):
        entries = []
        for key, page in list(self.pages.items()):
            with page.condition:
                if page.stopped or page.terminal:
                    continue
                item = {'lease_id': key, 'lease_token': page.lease['lease_token'],
                        'translations_revision': page.translations['revision'] if page.translations else None}
                # v3 has no separate persistence phase on the wire.
                item['phase'] = 'render' if page.phase == 'persist' else page.phase
                entries.append(item)
        return entries

    def adopt(self, lease, server_time, sent_at):
        key = lease['lease_id']
        if key in self.pages:
            self.pages[key].update(lease, server_time, sent_at)
            return
        if lease['status'] == 'terminal':
            self.journal.remove('lease:' + key)
            return
        saved = self.journal.get('lease:' + key, {})
        # Registration revalidates every input descriptor and execution lease.
        public_lease = {k: v for k, v in lease.items() if k != 'input'}
        self.journal.put('lease:' + key, {**saved, 'lease': public_lease})
        page = Page(lease, server_time, sent_at)
        page.analysis = lease.get('analysis') or saved.get('analysis')
        page.completion = saved.get('completion')
        if page.stopped or page.completion:
            page.step = 'deliver'
            if page.completion:
                self.pipeline.resize_reservation(page, self.pipeline.delivery_reservation(page.completion))
        elif lease['config']['engine']['protocol_version'] != 3:
            self.pipeline.error(page, NodeFailure('PROTOCOL_MISMATCH'))
        self.pages[key] = page
        self.wake.set()

    def register(self):
        sent_at = time.monotonic()
        response = self.transport.post('/nodes/register', {'protocol_version': 3, 'control_channel': 'websocket',
            'engine_version': self.runtime.version, 'resource_id': self.local['resource_id'],
            'device': self.local['engine'].get('backend', 'cuda') + ':' + str(self.local['engine']['gpu']),
            'supported_languages': self.runtime.languages, 'ready': True, 'result_formats': ['overlay-v1', 'overlay-tiles-v1'],
            'supported_font_ids': self.runtime.supported_font_ids,
            'document_protocol': self.documents.protocol()})
        if response['protocol_version'] != 3:
            raise NodeFailure('PROTOCOL_MISMATCH')
        self.apply_config(response['config'])
        self.server_clock_offset = sent_at - timestamp(response['server_time'])
        outstanding = {lease['lease_id'] for lease in response['leases']}
        # A completed/reclaimed lease absent from registration can be discarded.
        for key in self.journal.leases():
            if key not in outstanding:
                self.journal.remove('lease:' + key)
        for lease in response['leases']:
            self.adopt(lease, response['server_time'], sent_at)
        self.last_heartbeat = sent_at
        self.connected('registration')
        if self.documents.protocol() == 2:
            self.documents.sync()

    def heartbeat(self):
        entries = self.lease_entries()
        sent_at = time.monotonic()
        response = self.transport.post(f'/nodes/{self.local["node_id"]}/heartbeat',
            {'config_version': self.config['version'], 'leases': entries})
        self.apply_config(response.get('config'))
        self.server_clock_offset = sent_at - timestamp(response['server_time'])
        for item in response['leases']:
            page = self.pages.get(item['lease_id'])
            if page:
                page.update(item, response['server_time'], sent_at)
        self.connected('heartbeat')

    def claim(self):
        key = 'claim' if self.journal.get('claim') else 'document-claim'
        pending = self.journal.get(key)
        if not pending:
            count = self.claim_capacity()
            documents = int(self.documents.protocol() == 2 and bool(self.documents.capacity()))
            if not self.config['enabled'] or not (count or documents) or self.stop.is_set():
                return False
            key = 'claim'
            pending = {'request_id': uuid4().hex, 'config_version': self.config['version'],
                       'count': count, 'document_count': documents}
            self.journal.put(key, pending)
        channel = 'document_count' in pending
        sent_at = time.monotonic()
        try:
            if channel:
                response = self.transport.request_assignment(pending)
            else:
                post = self.transport.document_post if key == 'document-claim' else self.transport.post
                response = post(f'/nodes/{self.local["node_id"]}/claim', pending)
        except ControlFailure as error:
            if error.code == 'NODE_CONFIG_CONFLICT':
                # The controller checks saved receipts before config conflicts.
                self.journal.remove(key)
                self.last_heartbeat = 0
            raise
        if response.get('request_id') != pending['request_id'] or channel and not response['leases']:
            raise NodeFailure('CONTROL_INVALID_RESPONSE')
        self.apply_config(response.get('config'))
        if channel:
            # Capacity can be offered long before dispatch. Calibrate pushed
            # deadlines from the most recent request/heartbeat, never offer time.
            sent_at = timestamp(response['server_time']) + self.server_clock_offset
        for lease in response['leases']:
            adopter = self.documents.adopt if key == 'document-claim' or lease.get('workflow_kind') == 'document' else self.adopt
            adopter(lease, response['server_time'], sent_at)
        if channel:
            self.transport.acknowledge(pending['request_id'])
        self.journal.remove(key)  # Only after every returned lease is durable.
        return bool(response['leases'])

    def retry(self, page, operation):
        while True:
            page.check()
            try:
                return operation()
            except ControlFailure as error:
                if error.status and error.status < 500 and error.status != 429:
                    if error.code in {'LEASE_EXPIRED', 'PAGE_DEADLINE_EXCEEDED', 'ASSET_EXPIRED'}:
                        raise NodeFailure('LEASE_STOPPED') from None
                    raise
                with page.condition:
                    page.condition.wait(.5)

    def input_bytes(self, page):
        metadata = page.lease.get('input')
        if not metadata:
            raise ControlFailure('CONTROL_INVALID_RESPONSE')
        partial = bytearray()  # Per-page memory, released when this download ends.
        data = self.retry(page, lambda: self.transport.download(page.lease['lease_id'],
            page.lease['lease_token'], metadata, page.check, partial=partial))
        return data, metadata

    def deliver(self, page, body):
        page.phase = 'deliver'
        key = page.lease['lease_id']
        if 'result' in body:
            data = self.journal.output('lease:' + key)
            reply = self.retry(page, lambda: self.transport.deliver(key, body, data, page.check))
        else:
            reply = self.retry(page, lambda: self.transport.post(f'/leases/{key}/complete', body))
        if reply.get('status') != 'terminal':
            raise ControlFailure('CONTROL_INVALID_RESPONSE')
        return reply

    def reap(self):
        self.pipeline.tick()
        for key, page in list(self.pages.items()):
            if page.future or page.analysis_future:
                continue
            try:
                page.check()
            except NodeFailure:
                pass
            if not page.terminal and page.stopped:
                self.pipeline.retain_masks(page)
                if time.monotonic() >= page.next_stop:
                    page.next_stop = time.monotonic() + 1
                    page.step = 'deliver'
                    page.future = self.pipeline.delivery.submit(self.transport.post, f'/leases/{key}/complete',
                        {'lease_token': page.lease['lease_token'], 'error': {'code': 'LEASE_STOPPED'}})
                    page.future.add_done_callback(lambda _: self.wake.set())
                continue
            if page.terminal:
                self.pipeline.release(page)
                self.journal.remove('lease:' + key)
                del self.pages[key]
                self.request_claim()

    def close(self):
        self.stop.set()
        if self.transport is not None:
            self.transport.close_channel()
        if self.operations:
            self.operations.stopping()
        self.claim_pool.shutdown(wait=True)
        self.heartbeat_pool.shutdown(wait=True)
        for page in self.active_pages():
            page.stopped = True
        self.wake.set()
        self.channel_pool.shutdown(wait=True)
        self.documents.close_control()
        self.pipeline.close()
        self.documents.close()

    def listen(self):
        connected = False
        def updated(response):
            nonlocal connected
            if not connected:
                connected = True
                self.request_claim()
            self.channel_backoff = 1
            while not self.stop.is_set():
                try:
                    self.updates.put(response, timeout=.1)
                    self.wake.set()
                    return
                except Full:
                    continue
        self.transport.listen(self.stop, updated)

    def poll_control(self):
        if self.claim_future and self.claim_future.done():
            try:
                self.claim_future.result()
                self.network_log.succeeded('claim')
                self.next_claim = 0
            except Exception as error:
                if not isinstance(error, NodeFailure):
                    if self.journal.get('claim') or not self.journal.get('document-claim'):
                        raise
                    error = NodeFailure('DOCUMENT_WORKSPACE_UNAVAILABLE')
                self.network_log.failed('claim', error)
                self.next_claim = time.monotonic() + 1
            self.claim_future = None
        if self.heartbeat_future and self.heartbeat_future.done():
            try:
                self.heartbeat_future.result()
                self.network_log.succeeded('heartbeat')
            except NodeFailure as error:
                self.network_log.failed('heartbeat', error)
            self.heartbeat_future = None
        while True:
            try:
                response = self.updates.get_nowait()
            except Empty:
                break
            self.apply_config(response.get('config'))
            for item in response.get('leases', []):
                page = self.pages.get(item['lease_id']) or self.documents.pages.get(item['lease_id'])
                if page:
                    page.update(item)
            self.network_log.succeeded('channel')
        if self.channel_future and self.channel_future.done():
            try:
                self.channel_future.result()
            except NodeFailure as error:
                self.network_log.failed('channel', error)
            self.next_channel = time.monotonic() + self.channel_backoff
            self.channel_backoff = min(5, self.channel_backoff * 2)
            self.channel_future = None
        if not self.heartbeat_future and time.monotonic() - self.last_heartbeat >= self.config['heartbeat_seconds']:
            self.last_heartbeat = time.monotonic()
            self.heartbeat_future = self.heartbeat_pool.submit(self.heartbeat)
            self.heartbeat_future.add_done_callback(lambda _: self.wake.set())
        if (not self.stop.is_set() and time.monotonic() >= self.next_claim and self.claim_future is None
                and (self.journal.get('claim') or self.journal.get('document-claim') or self.claim_capacity() or
                     self.documents.protocol() == 2 and self.documents.capacity())):
            self.claim_future = self.claim_pool.submit(self.claim)
            self.claim_future.add_done_callback(lambda _: self.wake.set())
        if not self.channel_future and not self.stop.is_set() and time.monotonic() >= self.next_channel:
            self.channel_future = self.channel_pool.submit(self.listen)
            self.channel_future.add_done_callback(lambda _: self.wake.set())

    def run(self):
        try:
            delay = 1
            while not self.stop.is_set():
                if self.operations:
                    self.operations.pulse()
                try:
                    self.register()
                    break
                except ControlFailure as error:
                    self.network_log.failed('registration', error)
                    if error.status and error.status < 500 and error.status != 429:
                        raise
                    self.stop.wait(delay)
                    delay = min(30, delay * 2)
            self.next_claim = 0
            while not self.stop.is_set() or self.active_pages():
                if self.operations:
                    self.operations.pulse()
                try:
                    self.poll_control()
                    self.documents.poll_control()
                    self.reap()
                    if self.stop.is_set() and all(not p.future and not p.analysis_future for p in self.active_pages()):
                        break  # Keep uncertain deliveries in the journal for startup reconciliation.
                except NodeFailure as error:
                    self.network_log.failed('control', error)
                self.wake.wait(.02)
                self.wake.clear()
        finally:
            self.close()
