import argparse
from typing import Dict

from examples.classification import classification_example
from examples.regression import regression_example

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--example', required=True)
    
    return parser

def parse_arguments(parser: argparse.ArgumentParser) -> Dict[str, str]:
    args: argparse.Namespace = parser.parse_args()
    
    return vars(args)
    
def main():
    parser = init_parser()
    args = parse_arguments(parser)
    
    if args['example'] == 'classification':
        classification_example()
    else: 
        regression_example()
    
if __name__ == '__main__':
    main()