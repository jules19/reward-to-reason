"""The tower, as you've built it so far.

    python -m r2r.progress          # quick check (skips slow victory tests)
    python -m r2r.progress --full   # the whole examination

Each floor lights up when its module's tests pass. The quick check skips
the training-heavy "victory" tests; run the full exam (or the module's
own `pytest tests/test_moduleNN.py`) before declaring a floor truly
yours.
"""

import subprocess
import sys

from r2r.course import MODULE_DIRS, ROOT

FLOORS = [
    (10, "REASON", "net + search + self-play"),
    (9, "IMAGINATION", "minimax, MCTS"),
    (8, "THE MIRROR", "self-play"),
    (7, "THE BRAIN", "deep Q, generalization"),
    (6, "THE GRADIENT MACHINE", "backprop"),
    (5, "THE WALL", "instruments"),
    (4, "FORESIGHT", "Q-learning"),
    (3, "PLACE", "the experience map"),
    (2, "DOUBT", "exploration"),
    (1, "PREFERENCE", "the nudge rule"),
]


def check(number, full):
    test_file = ROOT / "tests" / f"test_module{number:02d}.py"
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-q",
           "--no-header", "-p", "no:cacheprovider"]
    if not full:
        cmd += ["-m", "not slow"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return result.returncode == 0


def main():
    full = "--full" in sys.argv
    print("Examining the tower" + (" (full exam)" if full else
          " (quick check; --full for the whole exam)") + "...\n")
    status = {}
    for number in sorted(MODULE_DIRS):
        status[number] = check(number, full)
        mark = "+" if status[number] else " "
        sys.stdout.write(f"  [{'#' if status[number] else '.'}] module {number:2d}\r")
        sys.stdout.flush()
    sys.stdout.write(" " * 30 + "\r")

    built = sum(status.values())
    for number, name, what in FLOORS:
        if status[number]:
            print(f"   ########  {name:<22} {what}")
        else:
            print(f"   ........  {name:<22} {what}")
    print(f"   ========  {'REWARD':<22} a number that sometimes goes up")
    print(f"\n   {built}/10 floors built.")
    if built == 0:
        print("   The tower awaits. Start at modules/01-the-coin/.")
    elif built < 10:
        nxt = min(n for n, ok in status.items() if not ok)
        print(f"   Next floor: modules/{MODULE_DIRS[nxt]}/")
    else:
        print("   The tower stands. Settle the bet: python -m r2r.play boss")


if __name__ == "__main__":
    main()
