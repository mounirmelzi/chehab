"""
Safe dataset cleaner for CHEHAB training data.

- Reads the input file READ-ONLY.
- Writes a NEW file (input name + ".cleaned") with only the well-formed
  lines, byte-identical to the original for every line that's kept.
- Writes a second file (input name + ".rejected") listing every line that
  was dropped and why, so you can eyeball what got removed.
- Never modifies, deletes, or renames the original file.

Uses the exact same parse_sexpr(...) check load_expressions() already uses,
so a line only gets dropped here if it would ALSO have been silently
skipped at training time. This just does it once instead of every run.

Usage (from RL/, inside chehabEnv):
    python clean_dataset.py fhe_rl/datasets/final_llm_dataset.txt

Output:
    fhe_rl/datasets/final_llm_dataset.txt.cleaned
    fhe_rl/datasets/final_llm_dataset.txt.rejected
"""
import sys
import os

try:
    from pytrs import parse_sexpr
except ImportError:
    print("Could not import parse_sexpr from pytrs.")
    print("Run this inside the chehabEnv conda env, from RL/ "
          "(same place you run `python -m fhe_rl`).")
    sys.exit(1)


def clean_dataset(input_path: str):
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    cleaned_path = input_path + ".cleaned"
    rejected_path = input_path + ".rejected"

    if os.path.exists(cleaned_path) or os.path.exists(rejected_path):
        print("Output file(s) already exist next to the input:")
        if os.path.exists(cleaned_path):
            print(f"  {cleaned_path}")
        if os.path.exists(rejected_path):
            print(f"  {rejected_path}")
        print("Refusing to overwrite — delete/rename them first if you want to re-run.")
        sys.exit(1)

    total = 0
    kept = 0
    rejected = 0

    # Read-only open on the source file.
    with open(input_path, "r") as src, \
         open(cleaned_path, "w") as out_clean, \
         open(rejected_path, "w") as out_rejected:

        for raw_line in src:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            # Preserve blank lines and comments as-is (load_expressions skips
            # them too, but we don't want to silently restructure the file).
            if not stripped or stripped.startswith("#"):
                out_clean.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            total += 1
            exp_str = stripped.split(":")[0].strip()

            try:
                parse_sexpr(exp_str)
            except Exception as e:
                rejected += 1
                out_rejected.write(f"{line}\n    -> {e}\n")
                continue

            kept += 1
            out_clean.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")

    print(f"Input file (untouched):  {input_path}")
    print(f"Cleaned copy written to: {cleaned_path}")
    print(f"Rejected-lines log:      {rejected_path}")
    print()
    print(f"Total expression lines checked: {total}")
    print(f"Kept (valid):                   {kept}")
    print(f"Rejected (malformed):           {rejected}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_dataset.py <path_to_dataset.txt>")
        sys.exit(1)
    clean_dataset(sys.argv[1])