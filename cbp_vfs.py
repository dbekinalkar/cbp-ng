#!/usr/bin/env python3

import subprocess
import sys
import tempfile


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "Usage: ./cbp_vfs.py <cbp_exec> <tracefile> <name> <warminst> <siminst>",
            file=sys.stderr,
        )
        return 1

    cbp_exec, tracefile, name, warminst, siminst = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = f"{name}.out"
        tmp_out_file = f"{tmpdir}/{out_file}"

        with open(tmp_out_file, "w", encoding="utf-8") as f:
            cbp = subprocess.run(
                [cbp_exec, tracefile, name, warminst, siminst],
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
            )

        if cbp.returncode != 0:
            sys.stderr.write(cbp.stderr)
            return cbp.returncode

        with open(tmp_out_file, "r", encoding="utf-8") as f:
            cbp_summary = f.readline().strip()

        metrics = subprocess.run(
            [sys.executable, "predictor_metrics.py", tmpdir],
            capture_output=True,
            text=True,
            check=True,
        )

        cbp = subprocess.run(
            [sys.executable, "vfs.py", metrics.stdout.strip()],
            capture_output=True,
            text=True,
            check=True,
        )

    print(cbp_summary)
    print(cbp.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
