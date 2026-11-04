"""UAV CLI: angular selection, point detection and top-view segmentation."""
import argparse
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
    parser.add_argument('--checkpoint', required=True, help='SANSA universal checkpoint')
    parser.add_argument('--sam2-checkpoint', required=True, help='Original SAM2 Hiera Large base weights')
    parser.add_argument('--supports', required=True, help='Per-class support manifest JSON')
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
    select = commands.add_parser('select', help='Export angularly selected frames without loading a model')
    _video_options(select)
    detect = commands.add_parser('detect', help='Segment point-class objects in selected video views')
    _video_options(detect)
    _model_options(detect)
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
        from .pipeline import detect_video, segment_top_view, select_video_frames
        if args.command != 'segment-top-view':
            selection = dict(mode=args.selection_mode, views=args.views, start_s=args.start_seconds,
                             end_s=args.end_seconds, angle_offset=args.angle_offset,
                             tolerance=args.angle_tolerance, centre=args.orbit_centre,
                             pitches=args.pitches, pitch_tolerance=args.pitch_tolerance)
        if args.command == 'select':
            result = select_video_frames(args.video, args.srt, args.output, **selection)
        else:
            if args.min_area_px < 1:
                raise ValueError('--min-area-px must be positive')
            geometry = 'Point' if args.command == 'detect' else 'Polygon'
            classes = classes_for(geometry, args.classes)
            from .models.sansa import SansaBackend
            backend = SansaBackend(args.sansa_repo, args.checkpoint, args.sam2_checkpoint,
                                   args.supports, args.device, args.image_size, args.threshold)
            if args.command == 'detect':
                result = detect_video(args.video, args.srt, args.output, backend, classes,
                                      min_area_px=args.min_area_px, **selection)
            else:
                result = segment_top_view(args.image, args.output, backend, classes,
                                          projection_json=args.projection_json,
                                          min_area_px=args.min_area_px)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f'UAV error: {error}\n')
    print(result)


if __name__ == '__main__':
    main()
