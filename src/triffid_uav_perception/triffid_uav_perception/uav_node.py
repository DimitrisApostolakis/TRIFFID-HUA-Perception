"""UAV CLI: angular selection, point detection and top-view segmentation."""
import argparse
import json
import logging

from .ontology import classes_for


def _video_options(parser):
    parser.add_argument('video', help='MP4/MOV constant-frame-rate drone video')
    parser.add_argument('--srt', help='DJI SRT; defaults to video basename .srt/.SRT')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--selection-mode', choices=['orbit', 'heading'], default='orbit',
                        help='orbit: GPS bearing about circle centre; heading: gimbal yaw')
    parser.add_argument('--views', type=int, default=8, help='Views per pitch, default 8')
    parser.add_argument('--start-seconds', type=float, default=0)
    parser.add_argument('--end-seconds', type=float, help='Exclusive end of one circle')
    parser.add_argument('--angle-offset', type=float, default=0)
    parser.add_argument('--angle-tolerance', type=float, default=12)
    parser.add_argument('--orbit-centre', type=float, nargs=2, metavar=('LAT', 'LON'),
                        help='Circle centre; otherwise estimated from GPS bounds in time window')
    parser.add_argument('--pitches', type=float, nargs='+',
                        help='Optional gimbal targets, e.g. --views 4 --pitches -90 -45')
    parser.add_argument('--pitch-tolerance', type=float, default=10)


def _model_options(parser):
    parser.add_argument('--backend', choices=['sansa'], default='sansa')
    parser.add_argument('--sansa-repo', required=True, help='Local official SANSA checkout')
    parser.add_argument('--checkpoint', required=True, help='SANSA Generalist/Universal checkpoint (channel factor 0.8)')
    parser.add_argument('--sam2-checkpoint', required=True, help='Original SAM2 Hiera Large base weights')
    parser.add_argument('--support-bank', '--supports', dest='support_bank', default='uav_supports',
                        help='Shared bank directory or manifest JSON (default: uav_supports)')
    parser.add_argument('--shots', type=int, choices=[1, 5], default=5,
                        help='Use first 1 or all 5 bank examples per class (default: 5)')
    parser.add_argument('--device', default='cuda', help='cuda, cuda:0, or cpu')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-area-px', type=int, default=1,
                        help='Minimum connected-component area at original image resolution')
    parser.add_argument('--classes', help='Optional comma-separated names/IDs within the task class list')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--verbose', action='store_true')
    commands = parser.add_subparsers(dest='command', required=True)
    bank = commands.add_parser('bank', help='Manage the shared five-example-per-class support bank')
    bank_commands = bank.add_subparsers(dest='bank_command', required=True)
    initialize = bank_commands.add_parser('init', help='Create UAV class entries with 5 empty slots each')
    initialize.add_argument('--bank', default='uav_supports', help='Bank directory to create')
    validate = bank_commands.add_parser('validate', help='Check all five examples per requested class')
    validate.add_argument('--bank', default='uav_supports', help='Bank directory or manifest JSON')
    validate.add_argument('--classes', help='Optional class names/IDs; defaults to all UAV classes')
    add = bank_commands.add_parser('add', help='Add a labelled image/mask pair to one unoccupied slot')
    add.add_argument('--bank', default='uav_supports')
    add.add_argument('--class', dest='class_name', required=True)
    add.add_argument('--slot', type=int, choices=range(1, 6), required=True)
    add.add_argument('--image', required=True)
    add.add_argument('--mask', required=True)
    add.add_argument('--foreground-value', type=int, help='Target label value for indexed multiclass masks')
    select = commands.add_parser('select', help='Export angularly selected frames without loading a model')
    _video_options(select)
    detect = commands.add_parser('detect', help='Segment point-class objects in selected video views')
    _video_options(detect)
    _model_options(detect)
    frames = commands.add_parser('detect-frames', help='Infer existing frames from selected_frames.json, without reading MP4')
    frames.add_argument('manifest', help='Previously exported selected_frames.json')
    frames.add_argument('--output', required=True)
    _model_options(frames)
    track = commands.add_parser('track-frames', help='Track existing point detections; export unique Unity points')
    track.add_argument('observations', help='Existing point_detections.json; detector is not rerun')
    track.add_argument('--output', required=True, help='New tracking output directory')
    track.add_argument('--confirmation-hits', type=int, default=2)
    track.add_argument('--min-score', type=float, default=.5)
    track.add_argument('--min-area-px', type=int, default=16)
    track.add_argument('--memory-seconds', type=int, default=900)
    track.add_argument('--max-identities', type=int, default=4096)
    track.add_argument('--accept-unscored-masks', action='store_true',
                       help='Accept binary-mask proposals without fabricating model confidence')
    track.add_argument('--no-overlays', action='store_true')
    track.add_argument('--preview-fps', type=float, default=1)
    top = commands.add_parser('segment-top-view', help='Segment polygon classes in a supplied top-view image')
    top.add_argument('image')
    top.add_argument('--output', required=True)
    top.add_argument('--projection-json', help='Optional Unity top-view projection context to carry into output')
    _model_options(top)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    try:
        if args.command == 'track-frames':
            from .tracking import TrackingSettings, track_document
            from .tracking.masa import MasaImageTracker
            from .tracking.visualize import TrackingRenderer
            from .export import write_json
            from pathlib import Path
            import time
            settings = TrackingSettings(args.confirmation_hits, args.min_score, args.min_area_px,
                                        args.memory_seconds, args.max_identities,
                                        accept_unscored_masks=args.accept_unscored_masks)
            if args.preview_fps <= 0:
                raise ValueError('--preview-fps must be positive')
            started = time.perf_counter()
            tracker = MasaImageTracker(settings.memory_seconds)
            preparation_s = time.perf_counter() - started
            renderer = None if args.no_overlays else TrackingRenderer(args.output, args.preview_fps)
            try:
                report = track_document(args.observations, args.output, tracker,
                                        settings=settings, render=renderer)
            finally:
                if renderer:
                    renderer.close()
            report['tracker_preparation_s'] = preparation_s
            write_json(Path(args.output)/'timings.json', report)
            print(json.dumps({key: value for key, value in report.items() if key != 'timings'}, indent=2))
            return
        if args.command == 'bank':
            from .supports import SupportBank, resolve_class, add_example, initialize_bank
            from .ontology import CLASSES
            if args.bank_command == 'init':
                print(initialize_bank(args.bank))
            elif args.bank_command == 'add':
                print(add_example(args.bank, args.class_name, args.slot, args.image,
                                  args.mask, args.foreground_value))
            else:
                classes = tuple(dict.fromkeys(resolve_class(v) for v in args.classes.split(','))) if args.classes else CLASSES
                report = SupportBank(args.bank).validate(classes)
                print(json.dumps(report, indent=2))
                if not report['valid']:
                    parser.exit(2)
            return
        from .pipeline import detect_video, detect_frames, segment_top_view, select_video_frames
        if args.command in ('select', 'detect'):
            selection = dict(mode=args.selection_mode, views=args.views, start_s=args.start_seconds,
                             end_s=args.end_seconds, angle_offset=args.angle_offset,
                             tolerance=args.angle_tolerance, centre=args.orbit_centre,
                             pitches=args.pitches, pitch_tolerance=args.pitch_tolerance)
        if args.command == 'select':
            result = select_video_frames(args.video, args.srt, args.output, **selection)
        else:
            if args.min_area_px < 1:
                raise ValueError('--min-area-px must be positive')
            geometry = 'Point' if args.command in ('detect', 'detect-frames') else 'Polygon'
            classes = classes_for(geometry, args.classes)
            from .supports import SupportBank
            supports = SupportBank(args.support_bank).select(classes, shots=args.shots)
            from .models.sansa import SansaBackend
            backend = SansaBackend(args.sansa_repo, args.checkpoint, args.sam2_checkpoint,
                                   args.device, args.image_size, args.threshold)
            if args.command == 'detect':
                result = detect_video(args.video, args.srt, args.output, backend, classes,
                                      supports=supports, min_area_px=args.min_area_px, **selection)
            elif args.command == 'detect-frames':
                result = detect_frames(args.manifest, args.output, backend, classes,
                                       supports=supports, min_area_px=args.min_area_px)
            else:
                result = segment_top_view(args.image, args.output, backend, classes,
                                          supports=supports, projection_json=args.projection_json,
                                          min_area_px=args.min_area_px)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f'UAV error: {error}\n')
    print(result)


if __name__ == '__main__':
    main()
