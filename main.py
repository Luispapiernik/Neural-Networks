from argparse import ArgumentParser

from neural_networks.use_cases.logical_and import learn_logical_and


def execute(args):
    learn_logical_and()


def main():
    parser = ArgumentParser()

    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
