"""UAV ontology with original TRIFFID IDs and separate human-role labels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class UavClass:
    id: int
    name: str
    geometry: str
    source_ids: tuple[int, ...]


_POINTS = {
    6: 'first responder', 43: 'military personnel',
    7: 'destroyed vehicle', 13: 'window', 16: 'debris', 21: 'civilian vehicle',
    23: 'citizen', 26: 'boat', 30: 'excavator', 31: 'door', 37: 'ambulance',
    38: 'fire truck', 47: 'crane', 48: 'stairs', 50: 'furniture', 51: 'tank',
    56: 'police vehicle', 58: 'army vehicle',
}
_POLYGONS = {
    0: 'water', 2: 'green tree', 4: 'flame', 5: 'smoke', 11: 'green plant',
    14: 'building', 15: 'destroyed building', 18: 'dirt road', 19: 'dry tree',
    22: 'road', 24: 'green grass', 27: 'pavement', 28: 'dry grass', 32: 'mud',
    36: 'burnt tree', 41: 'tower', 42: 'silo', 44: 'burnt grass',
    49: 'dry plant', 57: 'burnt plant',
}
CLASSES = tuple(sorted(
    (UavClass(i, name, geometry, (i,))
     for geometry, mapping in [('Point', _POINTS), ('Polygon', _POLYGONS)]
     for i, name in mapping.items()), key=lambda item: item.id))
BY_ID = {item.id: item for item in CLASSES}
BY_NAME = {item.name: item for item in CLASSES}


def classes_for(geometry: str, requested: str | None = None) -> tuple[UavClass, ...]:
    """A CLI subset may narrow a task, never silently change its geometry."""
    available = tuple(c for c in CLASSES if c.geometry == geometry)
    if requested is None:
        return available
    chosen = []
    for value in requested.split(','):
        value = value.strip().lower()
        item = BY_ID.get(int(value)) if value.isdigit() else BY_NAME.get(value)
        if item is None or item.geometry != geometry:
            raise ValueError(f'{value!r} is not a UAV {geometry} class')
        if item not in chosen:
            chosen.append(item)
    if not chosen:
        raise ValueError('At least one class must be selected')
    return tuple(chosen)
