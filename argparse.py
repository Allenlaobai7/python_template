import argparse

def main():
    parser = argparse.ArgumentParser(description='Featurize instances for ML classification.')
    parser.add_argument('-i', '--instances', nargs='*', default=[], metavar='<instances>', # collect into a list
                        help='List of instance files to featurize.')
    parser.add_argument('-o', '--output',  metavar='<features_uri>',
                        help='Save featurized instances here.')
    parser.add_argument('-s', '--settings', dest='settings_uri', metavar='<settings_uri>',
                        help='Settings file to configure models.')
    parser.add_argument('-p', '--parser_output', metavar='<featurizer>',
                        help='Fit instances and save featurizer model file here.')
    parser.add_argument('--foo', action='store_const', const=42) # store constant optional arguments
    parser.add_argument('--foo', action='store_true')     # store boolean
    A = parser.parse_args()

# nargs: number of arguments,
# --name: optional arguments
# name: positional arguments
# choices = ['a','b']: if not in list will return error

