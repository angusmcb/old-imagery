#!/usr/bin/env python3
"""Regenerate src/old_imagery/_descriptors/*.desc from the sources in proto/.

The package loads serialized ``FileDescriptorProto`` blobs at import time rather
than shipping ``protoc``-generated Python, so there is no build step for
installers. This script is what produces those blobs.

    pip install grpcio-tools
    python tools/regen_descriptors.py [--check]

``--check`` regenerates into a temporary directory and fails if the result
differs from what is committed, without touching the tree.

Every regeneration is compared field-by-field against the committed descriptor:
any message, field number, type, label or enum value that the current
descriptor has and the new one does not is a hard error. Purely additive
changes (Google adding fields to dbroot_v2) are reported and allowed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "proto"
DESC_DIR = ROOT / "src" / "old_imagery" / "_descriptors"
NAMES = ("dbroot_v2", "quadtreeset")


def compile_one(name: str, out_dir: Path) -> Path:
    """protoc the named .proto into a single serialized FileDescriptorProto."""
    fds_path = out_dir / f"{name}.fds"
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--descriptor_set_out={fds_path}",
        str(PROTO_DIR / f"{name}.proto"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"protoc failed for {name}.proto:\n{proc.stderr}")

    fds = descriptor_pb2.FileDescriptorSet.FromString(fds_path.read_bytes())
    if len(fds.file) != 1:
        sys.exit(f"{name}.proto produced {len(fds.file)} files, expected 1")

    desc_path = out_dir / f"{name}.desc"
    desc_path.write_bytes(fds.file[0].SerializeToString())
    fds_path.unlink()
    return desc_path


# -- compatibility checking ------------------------------------------------
def index(fdp: descriptor_pb2.FileDescriptorProto):
    """Flatten to {message_full_name: {field_number: (name, type, label, type_name)}}."""
    messages: dict[str, dict[int, tuple]] = {}
    enums: dict[str, dict[str, int]] = {}

    def walk(msgs, prefix: str) -> None:
        for m in msgs:
            full = f"{prefix}.{m.name}"
            messages[full] = {
                f.number: (f.name, f.type, f.label, f.type_name) for f in m.field
            }
            for e in m.enum_type:
                enums[f"{full}.{e.name}"] = {v.name: v.number for v in e.value}
            walk(m.nested_type, full)

    walk(fdp.message_type, fdp.package)
    for e in fdp.enum_type:
        enums[f"{fdp.package}.{e.name}"] = {v.name: v.number for v in e.value}
    return messages, enums


def compare(old_bytes: bytes, new_bytes: bytes, label: str) -> list[str]:
    old, old_enums = index(descriptor_pb2.FileDescriptorProto.FromString(old_bytes))
    new, new_enums = index(descriptor_pb2.FileDescriptorProto.FromString(new_bytes))
    problems: list[str] = []

    for msg in sorted(set(old) - set(new)):
        problems.append(f"{label}: message {msg} missing from regenerated descriptor")
    for msg in sorted(set(new) - set(old)):
        print(f"  [added] message {msg}")

    for msg in sorted(set(old) & set(new)):
        old_fields, new_fields = old[msg], new[msg]
        for num, (name, typ, lab, tname) in sorted(old_fields.items()):
            if num not in new_fields:
                problems.append(f"{label}: {msg} field #{num} ({name}) missing")
                continue
            n_name, n_typ, n_lab, n_tname = new_fields[num]
            if (typ, lab, tname) != (n_typ, n_lab, n_tname):
                problems.append(
                    f"{label}: {msg}.{name} (#{num}) changed shape: "
                    f"old=(type {typ}, label {lab}, {tname or '-'}) "
                    f"new=(type {n_typ}, label {n_lab}, {n_tname or '-'})"
                )
            elif n_name != name:
                problems.append(f"{label}: {msg} #{num} renamed {name!r} -> {n_name!r}")
        for num in sorted(set(new_fields) - set(old_fields)):
            print(f"  [added] {msg} field #{num} ({new_fields[num][0]})")

    for e in sorted(set(old_enums) - set(new_enums)):
        problems.append(f"{label}: enum {e} missing from regenerated descriptor")
    for e in sorted(set(old_enums) & set(new_enums)):
        for vname, vnum in old_enums[e].items():
            if new_enums[e].get(vname) != vnum:
                problems.append(
                    f"{label}: enum {e}.{vname} was {vnum}, now {new_enums[e].get(vname)}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed descriptors match proto/ without writing",
    )
    args = parser.parse_args()

    problems: list[str] = []
    stale: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name in NAMES:
            print(f"{name}.proto")
            new_path = compile_one(name, tmp_dir)
            new_bytes = new_path.read_bytes()

            committed = DESC_DIR / f"{name}.desc"
            if committed.exists():
                problems += compare(committed.read_bytes(), new_bytes, name)
                if committed.read_bytes() != new_bytes:
                    stale.append(name)
            else:
                print("  [new] no committed descriptor to compare against")

            if not args.check and not problems:
                shutil.copyfile(new_path, committed)

        if problems:
            print(f"\n!! {len(problems)} incompatibility(s) -- nothing written:")
            for p in problems:
                print("   -", p)
            return 1

        if args.check:
            if stale:
                print(f"\n!! out of date with proto/: {', '.join(stale)}")
                print("   run: python tools/regen_descriptors.py")
                return 1
            print("\nOK: committed descriptors match proto/")
        else:
            print(f"\nOK: wrote {len(NAMES)} descriptor(s) to {DESC_DIR}")
            print("     no message, field, type, label or enum value was lost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
