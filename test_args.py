import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='You can add a description here')

    parser.add_argument('-- experiment', help='Do you want to "train" or "inference"', required=True)
    parser.add_argument('--name', help='Your name', required=True)
    args = parser.parse_args()
    print(args.name)