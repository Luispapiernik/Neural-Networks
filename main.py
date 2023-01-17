from argparse import ArgumentParser

from neural_networks.use_cases.logistic_perceptron import (
    learn_logical_and, learn_logical_or, learn_logical_xor
)
from neural_networks.use_cases.shallow_neural_network import learn_cuadratic_function


def execute(args):
    # print("=" * 30)
    # learn_logical_and()
    # print("=" * 30)
    # learn_logical_or()
    # print("=" * 30)
    # learn_logical_xor()
    learn_cuadratic_function()


def main():
    parser = ArgumentParser()

    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
