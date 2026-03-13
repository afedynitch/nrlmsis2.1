"""Performance benchmark: Python NRLMSIS 2.1 vs Fortran reference.

Usage:
    python tests/bench_performance.py [--fortran]

    --fortran  Compile and time the Fortran reference binary (requires gfortran).

Scenarios
---------
A  Fixed location/time, 200 altitudes → ProfileCache hot, measures per-altitude overhead.
   Uses calc_altitude_array() (vectorized) and scalar calc() loop for comparison.
B  200 different locations from the validation test inputs → no cache benefit.
   Optionally compared against compiled Fortran.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nrlmsis import NRLMSIS21

DATA_DIR = Path(__file__).parent / 'data'
F90_DIR = Path(__file__).parent.parent / 'f90original'
F90_EXE = F90_DIR / 'msis_test.exe'

FORTRAN_SOURCES = [
    'msis_constants.F90',
    'msis_utils.F90',
    'msis_init.F90',
    'msis_gfn.F90',
    'msis_tfn.F90',
    'msis_dfn.F90',
    'msis_calc.F90',
    'msis_gtd8d.F90',
    'msis2.1_test.F90',
]


def load_test_cases():
    inputs = []
    with open(DATA_DIR / 'msis2.1_test_in.txt') as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            iyd = int(parts[0])
            sec = float(parts[1])
            alt = float(parts[2])
            glat = float(parts[3])
            glong = float(parts[4])
            f107a = float(parts[5])
            f107 = float(parts[6])
            ap = float(parts[7])
            ap7 = np.array([ap] * 7)
            inputs.append((iyd, sec, alt, glat, glong, f107a, f107, ap7))
    return inputs


def compile_fortran() -> bool:
    if not F90_DIR.exists():
        print(f'  Fortran source directory not found: {F90_DIR}')
        return False
    sources = [str(F90_DIR / s) for s in FORTRAN_SOURCES]
    missing = [s for s in sources if not Path(s).exists()]
    if missing:
        print(f'  Missing Fortran sources: {missing}')
        return False
    cmd = (
        ['gfortran', '-O3', '-march=native', '-ffast-math', '-DDBLE',
         '-o', str(F90_EXE)] + sources
    )
    print(f'  Compiling: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(F90_DIR))
    if result.returncode != 0:
        print(f'  Compile failed:\n{result.stderr}')
        return False
    print('  Compile OK')
    return True


def time_fortran(n_repeat: int = 5) -> float | None:
    if not F90_EXE.exists():
        return None
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = subprocess.run([str(F90_EXE)], capture_output=True, cwd=str(F90_DIR))
        t1 = time.perf_counter()
        if result.returncode != 0:
            return None
        times.append(t1 - t0)
    return min(times)


def bench(label: str, fn, n_warmup: int = 2, n_rep: int = 10):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    print(f'  {label}')
    print(f'    min={arr.min()*1e6:.1f} µs  median={np.median(arr)*1e6:.1f} µs  '
          f'mean={arr.mean()*1e6:.1f} µs  (n={n_rep})')
    return arr.min()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fortran', action='store_true',
                        help='Compile and benchmark Fortran reference')
    args = parser.parse_args()

    cases = load_test_cases()
    n = len(cases)
    print(f'Loaded {n} test cases\n')

    model = NRLMSIS21()
    ap_fixed = np.array([4.0] * 7)

    # ------------------------------------------------------------------ #
    # Scenario A: fixed location, 200 altitudes — ProfileCache hot        #
    # ------------------------------------------------------------------ #
    print('=' * 60)
    print('Scenario A: fixed location/time, 200 altitudes (cache hot)')
    print('=' * 60)
    z_200 = np.linspace(130.0, 800.0, 200)

    # Warm up profile cache
    model.calc(172, 43200, 200.0, 45.0, 0.0, 150.0, 148.0, ap_fixed)

    # Scalar loop
    def run_scalar_loop():
        for z in z_200:
            model.calc(172, 43200, float(z), 45.0, 0.0, 150.0, 148.0, ap_fixed)

    t_scalar = bench('calc() scalar loop (200 calls)', run_scalar_loop, n_warmup=3, n_rep=20)

    # Vectorized array
    def run_array():
        model.calc_altitude_array(172, 43200, z_200, 45.0, 0.0, 150.0, 148.0, ap_fixed)

    t_array = bench('calc_altitude_array() vectorized', run_array, n_warmup=3, n_rep=20)

    print(f'  Speedup (array vs scalar loop): {t_scalar / t_array:.1f}×\n')

    # ------------------------------------------------------------------ #
    # Scenario B: 200 different locations (validation cases), no cache    #
    # ------------------------------------------------------------------ #
    print('=' * 60)
    print('Scenario B: 200 different locations (no profile cache reuse)')
    print('=' * 60)

    model_b = NRLMSIS21()

    def run_multi_location():
        for iyd, sec, alt, glat, glong, f107a, f107, ap7 in cases:
            model_b.gtd8d(iyd, sec, alt, glat, glong, 0.0, f107a, f107, ap7)

    t_python = bench(f'gtd8d() {n} cases', run_multi_location, n_warmup=2, n_rep=10)
    print(f'  Per-call: {t_python / n * 1e6:.1f} µs')

    # Fortran comparison
    if args.fortran:
        print()
        print('Fortran benchmark:')
        if not F90_EXE.exists():
            compiled = compile_fortran()
        else:
            compiled = True

        if compiled:
            t_fort = time_fortran(n_repeat=20)
            if t_fort is not None:
                print(f'  Fortran {n} cases: {t_fort * 1e6:.1f} µs total, '
                      f'{t_fort / n * 1e6:.3f} µs/call')
                print(f'  Python/Fortran ratio: {t_python / t_fort:.1f}×')
            else:
                print('  Fortran run failed')


if __name__ == '__main__':
    main()
