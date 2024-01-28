import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="AutoClues")

    parser.add_argument(
        "-s",
        "--scenario",
        nargs="?",
        type=str,
        required=False,
        help="path to the scenario to execute",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        nargs="+",
        type=str,
        required=False,
        help="steps of the pipeline to execute",
    )

    parser.add_argument(
        "-r",
        "--result_path",
        nargs="?",
        type=str,
        required=False,
        help="path where put the results",
    )

    parser.add_argument(
        "-f",
        "--file_name",
        nargs="?",
        type=str,
        required=False,
        help="name of the input file name",
    )

    parser.add_argument(
        "-o",
        "--optimization_approach",
        nargs="?",
        type=str,
        required=False,
        help="The kind of optimization to execute",
    )

    parser.add_argument(
        "-sp",
        "--space_path",
        nargs="?",
        type=str,
        required=False,
        help="The kind of optimization to execute",
    )

    args = parser.parse_args()

    return args
