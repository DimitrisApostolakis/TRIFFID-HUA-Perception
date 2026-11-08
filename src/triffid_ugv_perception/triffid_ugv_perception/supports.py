"""Shared, model-independent UGV support bank with up to 16 examples per class.

The bank owns example identity and validation. The pipeline selects an immutable
episode; each backend only performs its own image/mask tensor preprocessing.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .io import write_json, read_rgb, write_rgb
from .ontology import BY_ID, BY_NAME, CLASSES

BANK_SCHEMA = 'triffid.ugv.support-bank.v1'
EXAMPLES_PER_CLASS = 16


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_class(value):
    value = str(value).strip().lower()
    item = BY_ID.get(int(value)) if value.isdigit() else BY_NAME.get(value)
    if item is None:
        raise ValueError(f'Unknown UGV class: {value!r}')
    return item


@dataclass(frozen=True)
class Support:
    image: Path
    mask: Path
    foreground_value: int | None = None
    example_id: str = ''

    def load(self):
        if not self.image.is_file() or not self.mask.is_file():
            raise ValueError(f'{self.example_id}: missing image/mask: {self.image}, {self.mask}')
        rgb = read_rgb(self.image)
        mask = cv2.imread(str(self.mask), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.shape != rgb.shape[:2]:
            raise ValueError(f'Support mask must be single-channel and match image: {self.mask}')
        foreground = mask > 0 if self.foreground_value is None else mask == self.foreground_value
        if not foreground.any():
            raise ValueError(f'Support mask has no foreground: {self.mask}')
        return rgb, foreground.astype(np.uint8)


@dataclass(frozen=True)
class SupportSelection:
    """Same ordered example identities for any backend, independent of its tensors."""
    bank_name: str
    manifest_sha256: str
    shots: int | str
    examples: tuple[tuple[int, tuple[Support, ...]], ...]
    fingerprints: tuple[tuple[str, str, str], ...]
    capacity: int = EXAMPLES_PER_CLASS

    def for_class(self, item):
        for class_id, examples in self.examples:
            if class_id == item.id:
                return examples
        raise ValueError(f'{item.name}: class is absent from selected supports')

    def require_classes(self, classes):
        if {item.id for item in classes} != {key for key, _ in self.examples}:
            raise ValueError('Support selection must match the requested model classes')

    def metadata(self):
        hashes = {key: (image, mask) for key, image, mask in self.fingerprints}
        return {
            'schema_version': BANK_SCHEMA, 'manifest': self.bank_name,
            'manifest_sha256': self.manifest_sha256, 'shots': self.shots,
            'selection_policy': 'first_k_in_manifest_order',
            'examples_per_class': self.capacity,
            'classes': {BY_ID[class_id].name: [
                {'example_id': support.example_id, 'image_sha256': hashes[support.example_id][0],
                 'mask_sha256': hashes[support.example_id][1],
                 'foreground_value': support.foreground_value}
                for support in examples] for class_id, examples in self.examples},
        }


class SupportBank:
    def __init__(self, location):
        self.path = Path(location).resolve()
        if self.path.is_dir():
            self.path /= 'bank.json'
        self.data = json.loads(self.path.read_text(encoding='utf-8-sig'))
        # Legacy five-slot manifests remain readable for reproducibility.
        if not isinstance(self.data, dict) or self.data.get('schema_version') not in (
                BANK_SCHEMA,):
            raise ValueError('Unsupported support bank schema')
        if not isinstance(self.data.get('classes'), dict):
            raise ValueError('Support bank must contain a classes object')
        self.capacity=self.data.get('examples_per_class',5)
        if type(self.capacity) is not int or self.capacity not in (5,EXAMPLES_PER_CLASS):
            raise ValueError('The UGV bank must define 16 slots per class (legacy 5 accepted)')
        unknown = set(self.data['classes']) - set(BY_NAME)
        if unknown:
            raise ValueError(f'Unknown support bank classes: {sorted(unknown)}')

    def for_class(self, item):
        entries = self.data['classes'].get(item.name)
        if not isinstance(entries, list) or len(entries) != self.capacity:
            raise ValueError(f'{item.name}: bank must store exactly {self.capacity} support slots')
        supports = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict) or not all(
                    isinstance(entry.get(k), str) and entry[k].strip() for k in ('image', 'mask')):
                raise ValueError(f'{item.name}: each support needs image and mask paths')
            value = entry.get('foreground_value')
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError('foreground_value must be a nonnegative integer')
            identifier = entry.get('id', f'example_{index:02d}')
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError('Support example IDs must be nonempty strings')
            supports.append(Support((self.path.parent / entry['image']).resolve(),
                                    (self.path.parent / entry['mask']).resolve(), value,
                                    f'{item.name}/{identifier}'))
        if len({s.example_id for s in supports}) != self.capacity:
            raise ValueError(f'{item.name}: support example IDs must be distinct')
        if len({s.image for s in supports}) != self.capacity:
            raise ValueError(f'{item.name}: provide distinct support images, not repeated paths')
        return tuple(supports)

    def _inspect(self, classes, shots=None):
        shots=self.capacity if shots is None else shots
        errors, fingerprints, valid = [], {}, 0
        for item in classes:
            try:
                examples = self.for_class(item)[:shots]
                if len(examples)!=shots:
                    raise ValueError(f'{item.name}: requested {shots} shots exceeds this manifest capacity {self.capacity}')
            except ValueError as error:
                errors.append(str(error))
                continue
            image_hashes = set()
            for support in examples:
                try:
                    support.load()  # Checks dimensions, channel count and foreground.
                    image_hash, mask_hash = _sha256(support.image), _sha256(support.mask)
                    if image_hash in image_hashes:
                        raise ValueError(f'{support.example_id}: duplicate image content within class')
                    image_hashes.add(image_hash)
                    fingerprints[support.example_id] = (image_hash, mask_hash)
                    valid += 1
                except (ValueError, OSError) as error:
                    errors.append(str(error))
        return errors, fingerprints, valid

    def validate(self, classes=CLASSES, shots=None):
        classes = tuple(classes)
        if shots is not None and (type(shots) is not int or not 1<=shots<=EXAMPLES_PER_CLASS):
            raise ValueError('shots must be an integer between 1 and 16')
        errors, _, valid = self._inspect(classes,shots)
        return {'valid': not errors, 'manifest': str(self.path),
                'class_count': len(classes), 'expected_examples': len(classes) * (self.capacity if shots is None else shots),
                'valid_examples': valid, 'errors': errors}

    def select(self, classes, shots=5):
        if type(shots) is not int or not 1<=shots<=EXAMPLES_PER_CLASS:
            raise ValueError('shots must be an integer between 1 and 16')
        classes = tuple(classes)
        if not classes or len({item.id for item in classes}) != len(classes):
            raise ValueError('Select a nonempty set of distinct UGV classes')
        errors, fingerprints, _ = self._inspect(classes, shots=shots)
        if errors:
            raise ValueError('Invalid support bank:\n' + '\n'.join(errors))
        examples = tuple((item.id, self.for_class(item)[:shots]) for item in classes)
        selected_hashes = tuple((s.example_id, *fingerprints[s.example_id])
                                for _, supports in examples for s in supports)
        return SupportSelection(self.path.name, _sha256(self.path), shots, examples, selected_hashes,self.capacity)

    def select_available(self, classes):
        """Select each class's valid populated prefix, up to the bank capacity."""
        classes=tuple(classes)
        if not classes or len({item.id for item in classes})!=len(classes):
            raise ValueError('Select a nonempty set of distinct UGV classes')
        selections=[]
        for item in classes:
            count=0
            for support in self.for_class(item):
                if not support.image.is_file() or not support.mask.is_file():break
                count+=1
            if not count:raise ValueError(f'{item.name}: no available supports')
            selections.append(self.select([item],count))
        if not selections:raise ValueError('Select a nonempty set of classes')
        return SupportSelection(self.path.name,_sha256(self.path),'available',
            tuple(row for selected in selections for row in selected.examples),
            tuple(row for selected in selections for row in selected.fingerprints),self.capacity)


def bank_template():
    return {'schema_version': BANK_SCHEMA, 'examples_per_class': EXAMPLES_PER_CLASS,
            'classes': {item.name: [
                {'id': f'example_{slot:02d}',
                 'image': f'{item.name.replace(" ", "_")}/{slot:02d}/image.png',
                 'mask': f'{item.name.replace(" ", "_")}/{slot:02d}/mask.png'}
                for slot in range(1, EXAMPLES_PER_CLASS + 1)] for item in CLASSES}}


def initialize_bank(directory):
    directory = Path(directory).resolve()
    manifest = directory / 'bank.json'
    if manifest.exists():
        raise ValueError(f'Bank already exists; existing examples were preserved: {manifest}')
    directory.mkdir(parents=True, exist_ok=True)
    # Empty slots are references, not fabricated example images or masks.
    for item in CLASSES:
        (directory / item.name.replace(' ', '_')).mkdir(exist_ok=True)
    return write_json(manifest, bank_template())


def expand_bank(location):
    """Expand a legacy manifest without changing or replacing any examples."""
    path=Path(location)
    if path.is_dir():path=path/'bank.json'
    bank=SupportBank(path)
    if bank.capacity==EXAMPLES_PER_CLASS:return path
    expanded=bank_template()
    for name,entries in bank.data['classes'].items():
        if len(entries)!=bank.capacity:raise ValueError(f'{name}: malformed legacy slots')
        expanded['classes'][name][:bank.capacity]=entries
    updated=dict(bank.data,examples_per_class=EXAMPLES_PER_CLASS,classes=expanded['classes'])
    raw=path.read_bytes()
    archive=path.parent/'history'/('capacity16_'+hashlib.sha256(raw).hexdigest()[:12])/'bank.json'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():archive.write_bytes(raw)
    return write_json(path,updated)


def add_example(location, class_name, slot, image, mask, foreground_value=None):
    """Normalize one supplied example into an unoccupied managed bank slot."""
    bank = SupportBank(location)
    item = resolve_class(class_name)
    if isinstance(slot, bool) or slot not in range(1, EXAMPLES_PER_CLASS + 1):
        raise ValueError('slot must be between 1 and 16')
    if slot>bank.capacity:
        raise ValueError('Expand the legacy bank to 16 slots before adding this slot')
    if bank.data['classes'][item.name][slot-1] != bank_template()['classes'][item.name][slot-1]:
        raise ValueError('Slot already contains a custom example; existing references were preserved')
    if foreground_value is not None and (not isinstance(foreground_value, int) or
                                          isinstance(foreground_value, bool) or foreground_value < 0):
        raise ValueError('foreground_value must be a nonnegative integer')
    rgb, binary = Support(Path(image), Path(mask), foreground_value, 'new example').load()
    target = bank.for_class(item)[slot - 1].image.parent
    if target.exists():
        raise ValueError(f'Slot already exists; no files overwritten: {target}')
    # Directory rename publishes image+mask together. Existing slots are never replaced.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='.adding-') as temporary:
        temporary = Path(temporary)
        write_rgb(temporary / 'image.png', rgb)
        if not cv2.imwrite(str(temporary / 'mask.png'), binary * 255):
            raise OSError('Could not save support mask')
        image_hash = _sha256(temporary / 'image.png')
        for support in bank.for_class(item):
            if support.image.is_file() and _sha256(support.image) == image_hash:
                raise ValueError(f'{item.name}: duplicate support image; choose a different example')
        temporary.rename(target)
    return target
