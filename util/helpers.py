from pathlib import Path
from typing import Literal


def check_output_exists(path_output: Path | str, file_exists: Literal['ask', 'delete', 'continue', 'exit']):
    if isinstance(path_output, str):
        path_output = Path(path_output)
    if path_output.exists():
        if file_exists == 'ask':
            size_mb = path_output.stat().st_size / (2**20)
            try:
                res = input(
                    f'file {path_output} exists ({size_mb:0.2f} MB), [d]elete , [c]ontinue or [N] abort? (d/c/N): ')
            except KeyboardInterrupt:
                exit()
            if res.lower() == 'd':
                path_output.unlink()
            elif res.lower() != 'c':
                exit()
        elif file_exists == 'delete':
            path_output.unlink()
        elif file_exists != 'continue':
            exit()
