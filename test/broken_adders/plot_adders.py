#!/usr/bin/env python3
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('input_file')
parser.add_argument('--output')

args = parser.parse_args()


with open(args.input_file) as f_in:
    hits = []
    cfg_inject_total = -1
    for line in f_in:
        line = line.strip()
        if line.startswith('# cfg_inject_total = '):
            line = line.removeprefix('# cfg_inject_total = ')
            cfg_inject_total = int(line)
            continue
        if not line or line.startswith('#'):
            continue
        hits.append(list(map(int, line.split())))

hits = np.array(hits) / cfg_inject_total
hits = 1 - hits
hits = np.where(hits == 0, np.nan, hits)

cmap = matplotlib.cm.get_cmap('Reds').copy() # type: ignore
cmap.set_bad('green')

op = Path(args.input_file)
plt.suptitle(f'Adder Sensitivity ({op.with_suffix("").name})')
plt.xlabel('Inject Column')
plt.xlabel('Inject Row')
plt.imshow(hits, cmap=cmap)
plt.tight_layout()

if args.output:
    plt.savefig(args.output, dpi=300, transparent=True, bbox_inches='tight')
else:
    plt.show()
