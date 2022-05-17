import argparse
import os

from .vamc import VariabilityAwareModelChecker, \
    DEFAULT_CPP_PATH, DEFAULT_SEAHORN_PATH, DEFAULT_Z3_PATH, DEFAULT_TIMEOUT, \
    DEFAULT_USE_NORMALIZED_PATH


def _extant_file(x: str) -> str:
    if not os.path.isfile(x):
        raise argparse.ArgumentTypeError(f"{x} is not a file")
    return x


parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="A variability-aware model checker "
                "using SeaHorn as the backend engine.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("path", help="Path to the C file to be checked",
                    metavar="FILE", type=_extant_file)
parser.add_argument("--features", help="Specify the feature variables",
                    nargs='*', required=True)
parser.add_argument("--cpp", help="Path to C Preprocessor",
                    metavar="PATH", default=DEFAULT_CPP_PATH)
parser.add_argument("--sea", help="Path to SeaHorn",
                    metavar="PATH", default=DEFAULT_SEAHORN_PATH)
parser.add_argument("--z3", help="Path to Z3",
                    metavar="PATH", default=DEFAULT_Z3_PATH)
parser.add_argument("--timeout", help="Set the timeout in seconds",
                    metavar='SECONDS', default=DEFAULT_TIMEOUT, type=int)
parser.add_argument("--normalize-path",
                    help="Enable if filepaths should be normalized",
                    action=f'store_{not DEFAULT_USE_NORMALIZED_PATH}'.lower())
parser.add_argument("--out", help="Path to the output directory")

namespace: argparse.Namespace = parser.parse_args()

for i, (feature_formula, input_formula) in enumerate(
        VariabilityAwareModelChecker(
            cpp_path=namespace.cpp, seahorn_path=namespace.sea,
            z3_path=namespace.z3, use_normalized_path=namespace.normalize_path,
            timeout=namespace.timeout,
        ).check(
            c_filepath=namespace.path, features=namespace.features,
            out_dir_path=namespace.out
        )
):
    print(f"Featured Counter Example")
    print("\tFeatures:", "\t", feature_formula.serialize())
    print("\tInputs:  ", "\t", input_formula.serialize())
    print("")
