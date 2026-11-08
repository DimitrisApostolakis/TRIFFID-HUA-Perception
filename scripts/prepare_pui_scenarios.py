"""Stage downloaded split bags without modifying originals (run on host)."""
import json
import os
from pathlib import Path
import re
import shutil
import zipfile
import yaml


SOURCE = Path('/mnt/d/Datasets/PUI/PUI - Rosbags')
OUTPUT = Path('/mnt/d/projects/triffid-hua-perception/data/PUI/scenarios')


def main():
    OUTPUT.mkdir(exist_ok=False)
    manifest = []
    for scenario in ('scenario_1', 'scenario_2'):
        folder = SOURCE / scenario
        for archive in sorted(folder.glob('*.zip')):
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError(f'Unsafe archive member: {relative}')
                    if member.is_dir():
                        continue
                    target = OUTPUT / scenario / 'bags' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(member) as source, target.open('xb') as destination:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                    manifest.append(dict(source=str(archive), member=member.filename,
                                         output=str(target.relative_to(OUTPUT)), mode='extracted'))
        for source in sorted(folder.iterdir()):
            match = re.fullmatch(r'(rosbag2_.+)_([0-9]+)(?:-[0-9]+)?\.(mcap|db3)', source.name)
            if not match:
                continue
            bag_name, index, extension = match.groups()
            target = OUTPUT / scenario / 'bags' / bag_name / f'{bag_name}_{index}.{extension}'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            try:
                os.link(source, target)
                mode = 'hardlink'
            except OSError:
                shutil.copy2(source, target)
                mode = 'copy'
            manifest.append(dict(source=str(source), output=str(target.relative_to(OUTPUT)), mode=mode))
        for metadata in (OUTPUT / scenario / 'bags').glob('*/metadata.yaml'):
            doc = yaml.safe_load(metadata.read_text())
            info = doc['rosbag2_bagfile_information']
            missing = [name for name in info['relative_file_paths'] if not (metadata.parent / name).is_file()]
            if missing:
                raise FileNotFoundError(missing)
            metadata.write_text(yaml.safe_dump(doc, sort_keys=False))
    (OUTPUT / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Staged {len(manifest)} files at {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
