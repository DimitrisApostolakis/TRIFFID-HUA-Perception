"""Manage the UGV-only bank without ROS, torch, model weights or a GPU."""
import argparse
import json
from .ontology import classes_for
from .supports import SupportBank, initialize_bank, add_example, expand_bank

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    initialize = commands.add_parser('init')
    initialize.add_argument('--bank', default='ugv_supports/PUI')
    expand=commands.add_parser('expand')
    expand.add_argument('--bank',default='ugv_supports/PUI')
    add = commands.add_parser('add')
    add.add_argument('--bank', default='ugv_supports/PUI')
    add.add_argument('--class', dest='class_name', required=True)
    add.add_argument('--slot', type=int, choices=range(1, 17), required=True)
    add.add_argument('--image', required=True)
    add.add_argument('--mask', required=True)
    add.add_argument('--foreground-value', type=int)
    validate = commands.add_parser('validate')
    validate.add_argument('--bank', default='ugv_supports/PUI')
    validate.add_argument('--classes', default='')
    validate.add_argument('--shots',type=int,choices=range(1,17),help='Validate this many slots; default checks full capacity')
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            print(initialize_bank(args.bank))
        elif args.command == 'expand':
            print(expand_bank(args.bank))
        elif args.command == 'add':
            print(add_example(args.bank, args.class_name, args.slot, args.image, args.mask, args.foreground_value))
        else:
            report = SupportBank(args.bank).validate(classes_for(args.classes),shots=args.shots)
            print(json.dumps(report, indent=2))
            if not report['valid']:
                parser.exit(2)
    except (ValueError, OSError) as error:
        parser.exit(2, f'UGV bank error: {error}\n')

if __name__ == '__main__':
    main()
