# Variability-Aware Model Checker

A variability-aware model checker using SeaHorn as the backend engine.

## Prerequisites:
1. Install [Python 3.10](https://www.python.org/downloads/).
2. Install required Python packages.
    ```bash
    pip install -r src/requirements.txt
    ```
3. Install [Seahorn (dev10)](https://github.com/seahorn/seahorn/tree/dev10).
4. Install [Z3](https://github.com/Z3Prover/z3/releases/tag/z3-4.8.9) (recommended version: 4.8.9) if Z3 hasn't been installed as required by Seahorn.

## Usage:
```
python src/main.py [-h] --features [FEATURES ...]
                   [--cpp PATH] [--sea PATH] [--z3 PATH] 
                   [--timeout SECONDS] [--wsl] FILE
```
- `FILE`: C source code file to be checked
- `--features [FEATURES ...]`: feature variables

Optional arguments:

- `-h`, `--help`: show the help message and exit
- `--cpp PATH`: C Preprocessor path (default: cpp)
- `--sea PATH`: SeaHorn path (default: sea)
- `--z3 PATH`: Z3 path (default: z3)
- `--timeout SECONDS`: timeout in seconds
- `--wsl`: enable if using Windows Subsystem for Linux (WSL)

## Examples:
```bash
python src/main.py examples/test1/merged.c --features FA FB FC
                   --sea "~/seahorn/build/run/bin/sea"
                   --z3 "~/z3-4.8.9/bin/z3"
```