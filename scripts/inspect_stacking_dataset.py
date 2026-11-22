#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_dataset import inspect_stacking_dataset
if __name__=="__main__": print("Stacking Dataset V1 integrity: PASSED\n",inspect_stacking_dataset(ROOT/"data/stacking_v1"))
