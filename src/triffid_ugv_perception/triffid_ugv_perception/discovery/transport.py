"""Legacy creation and full-Feature PUT/PATCH transports with explicit acknowledgements."""
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit, quote
from urllib.request import Request, urlopen

from .features import encode, location_ready


class PermanentDeliveryError(RuntimeError):
    pass


class CreateOnlyHTTPTransport:
    """Server must uniquely enforce (namespace, client_object_id).

    Merely sending an Idempotency-Key header does not establish that contract.
    The deployment flag is an operator assertion, not automatic verification.
    There is deliberately no proximity GET, PATCH or update operation.
    """
    def __init__(self, api_url, namespace, uniqueness_contract=False, timeout_s=5., opener=urlopen):
        if not uniqueness_contract:
            raise ValueError('API creation requires verified server-side uniqueness for namespace/client_object_id')
        if urlsplit(api_url).scheme not in ('http','https') or not 0 < timeout_s < 30:
            raise ValueError('Invalid API URL/timeout')
        self.url, self.namespace = api_url, namespace
        self.timeout_s, self.opener = timeout_s, opener

    def create(self, object_id, payload):
        feature = json.loads(payload)
        props = feature.get('properties',{})
        if (not location_ready(feature) or props.get('client_object_id') != object_id
                or props.get('namespace') != self.namespace):
            raise PermanentDeliveryError('Creation payload lacks valid location or matching UUID/namespace')
        request = Request(self.url,method='PUT',data=encode(self._create_body(feature)),
            headers={'Content-Type':'application/json','Idempotency-Key':f'{self.namespace}:{object_id}'})
        try:
            with self.opener(request,timeout=self.timeout_s) as response:
                raw = response.read(65537)
                if len(raw)>65536:
                    raise RuntimeError('API response exceeds limit')
                if response.status not in (200,201):
                    raise RuntimeError(f'API create returned {response.status}')
                data = json.loads(raw)
        except HTTPError as error:
            if 400 <= error.code < 500 and error.code not in (408,409,429):
                raise PermanentDeliveryError(f'API rejected create: HTTP {error.code}') from error
            # Ambiguous/conflict responses are never treated as acknowledged
            # without a remote ID. Retries use the exact same immutable key.
            raise
        remote_id = data.get('id') if isinstance(data,dict) else None
        if not isinstance(remote_id,(str,int)) or isinstance(remote_id,bool) or not str(remote_id) or len(str(remote_id))>256:
            raise RuntimeError('API did not acknowledge a remote object ID')
        return str(remote_id)

    def _create_body(self, feature):
        return dict(geometry=feature['geometry'], properties=feature['properties'])


class FeatureHTTPTransport(CreateOnlyHTTPTransport):
    """Create with a client ID; update a known POI with a full Feature body."""

    def _create_body(self, feature):
        return feature

    def update(self, object_id, remote_id, payload, operation_id):
        feature = json.loads(payload)
        props = feature.get('properties', {})
        if (not location_ready(feature) or props.get('client_object_id') != object_id
                or props.get('namespace') != self.namespace or not remote_id or not operation_id):
            raise PermanentDeliveryError('Update lacks valid location or matching object/POI identity')
        # Match Unity: URL and top-level id address the remote POI;
        # properties retain the persistent client identity.
        feature['type'], feature['id'] = 'Feature', str(remote_id)
        request = Request(self.url.rstrip('/')+'/'+quote(str(remote_id), safe=''),
            method='PATCH',data=encode(feature),headers={
                'Content-Type':'application/json',
                'Idempotency-Key':f'{self.namespace}:{object_id}:update:{operation_id}'})
        try:
            with self.opener(request,timeout=self.timeout_s) as response:
                raw = response.read(65537)
                if len(raw)>65536:
                    raise RuntimeError('API response exceeds limit')
                if response.status != 200:
                    raise RuntimeError(f'API update returned {response.status}')
                data = json.loads(raw)
        except HTTPError as error:
            if 400 <= error.code < 500 and error.code not in (408,409,429):
                raise PermanentDeliveryError(f'API rejected update: HTTP {error.code}') from error
            raise
        if not isinstance(data,dict) or data.get('success') is not True:
            raise RuntimeError('API did not acknowledge the update')
        return str(remote_id)
