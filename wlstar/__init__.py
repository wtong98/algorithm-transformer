"""
Weighted L* implementation from https://github.com/tech-srl/weighted_lstar
"""

from pathlib import Path
import sys

# resolve nested import issue
curr_path = Path(__file__).parent.resolve()
sys.path.append(str(curr_path))