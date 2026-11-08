"""Image-space identity tracking, independent of detection and georeferencing."""

from .session import TrackingSettings, track_document

__all__ = ['TrackingSettings', 'track_document']
