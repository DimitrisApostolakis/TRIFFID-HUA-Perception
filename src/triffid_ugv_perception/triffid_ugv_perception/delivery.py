"""Create-only delivery entrypoint; proximity upserts have been retired.

See discovery.transport for the server UUID uniqueness contract and
discovery.outbox for durable retries. No network is used on import.
"""
from .discovery.transport import CreateOnlyHTTPTransport, PermanentDeliveryError

__all__ = ['CreateOnlyHTTPTransport', 'PermanentDeliveryError']
