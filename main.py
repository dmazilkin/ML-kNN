import argparse
from typing import Dict

from examples.classification import classification_example
from examples.regression import regression_example

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--example', required=True)
    parser.add_argument('-t', '--train', required=True, type=int)
    parser.add_argument('-p', '--predict', required=True, type=int)
    parser.add_argument('-k', required=True, type=int)
    parser.add_argument('-w', '--weight', required=False, default='uniform')
    parser.add_argument('-m', '--metric', required=False, default='euclidean')
    
    return parser

def parse_arguments(parser: argparse.ArgumentParser) -> Dict[str, str]:
    args: argparse.Namespace = parser.parse_args()
    
    return vars(args)
    
def main():
    parser = init_parser()
    args = parse_arguments(parser)
    example = args['example']
    args.pop('example')
    
    if example == 'classification':
        classification_example(**args)
    else: 
        regression_example(**args)
    
if __name__ == '__main__':
    main()