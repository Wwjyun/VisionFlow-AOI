"""Meter-wheel DLL diagnostics: why `LSI8181_64.dll` cannot be loaded on this machine.

Windows reports the same "找不到指定的模組" for a missing DLL and for a missing *dependency* of a DLL,
which is not enough to fix anything on a machine whose files cannot be copied out. This module reads
the PE headers with the standard library only, so the application can report:

* every path the loader tried,
* the exact Windows loader error,
* the DLL's bitness (32-bit DLL in a 64-bit process is error 193, not 126),
* which imported DLLs cannot be resolved on this machine.

No Qt, no ctypes calls into the vendor DLL, no mutable module globals.
"""

from __future__ import annotations

import os
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from devices.lsi8181 import DLL_NAME, DLL_PATH_ENV, dll_candidates

_MACHINE_TYPES = {
    0x014C: "x86",
    0x8664: "x64",
    0x01C0: "ARM",
    0xAA64: "ARM64",
}
_API_SET_PREFIXES = ("api-ms-win-", "ext-ms-win-")
_PE32_MAGIC = 0x10B
_PE32_PLUS_MAGIC = 0x20B


@dataclass(frozen=True)
class PeImage:
    """The little we need from a PE file: its machine type and its direct imports."""

    machine: str
    imports: tuple[str, ...]

    @property
    def machine_code(self) -> int:
        return next((code for code, name in _MACHINE_TYPES.items() if name == self.machine), 0)


def read_pe_image(path: str | os.PathLike) -> PeImage | None:
    """Read machine type and imported DLL names; `None` when the file is not a readable PE image."""

    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    try:
        if data[:2] != b"MZ":
            return None
        (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
        if data[pe_offset : pe_offset + 4] != b"PE\0\0":
            return None
        machine_code, _sections, _stamp, _sym, _nsym, optional_size, _chars = struct.unpack_from(
            "<HHIIIHH", data, pe_offset + 4
        )
        optional = pe_offset + 24
        (magic,) = struct.unpack_from("<H", data, optional)
        if magic == _PE32_PLUS_MAGIC:
            directories = optional + 112
        elif magic == _PE32_MAGIC:
            directories = optional + 96
        else:
            return None
        import_rva, _import_size = struct.unpack_from("<II", data, directories + 8)
        imports = _read_imports(data, pe_offset, optional_size, import_rva) if import_rva else ()
        return PeImage(_MACHINE_TYPES.get(machine_code, f"0x{machine_code:04X}"), imports)
    except (struct.error, ValueError, IndexError):
        return None


def _sections(data: bytes, pe_offset: int, optional_size: int) -> list[tuple[int, int, int, int]]:
    (count,) = struct.unpack_from("<H", data, pe_offset + 6)
    base = pe_offset + 24 + optional_size
    sections = []
    for index in range(count):
        offset = base + index * 40
        _name, _vsize, vaddr, raw_size, raw_ptr = struct.unpack_from("<8sIIII", data, offset)
        sections.append((vaddr, vaddr + max(raw_size, 0), raw_ptr, raw_size))
    return sections


def _rva_to_offset(sections, rva: int) -> int | None:
    for vaddr, vend, raw_ptr, raw_size in sections:
        if vaddr <= rva < vend:
            delta = rva - vaddr
            return raw_ptr + delta if delta < raw_size else None
    return None


def _read_imports(data: bytes, pe_offset: int, optional_size: int, import_rva: int) -> tuple[str, ...]:
    sections = _sections(data, pe_offset, optional_size)
    names: list[str] = []
    descriptor = _rva_to_offset(sections, import_rva)
    if descriptor is None:
        return ()
    while descriptor + 20 <= len(data):
        original, _stamp, _forward, name_rva, _first = struct.unpack_from("<IIIII", data, descriptor)
        if original == 0 and name_rva == 0:
            break
        name_offset = _rva_to_offset(sections, name_rva)
        if name_offset is None:
            break
        end = data.find(b"\0", name_offset)
        if end < 0:
            break
        names.append(data[name_offset:end].decode("ascii", "replace"))
        descriptor += 20
    return tuple(names)


def missing_dependencies(path: str | os.PathLike, image: PeImage, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Imported DLLs that cannot be resolved next to the file, in the app folder, or in System32.

    Best effort by design: a delay-loaded or already-loaded module may still satisfy the import, so the
    caller reports this as a hint, never as a verdict.
    """

    env = os.environ if environ is None else environ
    root = env.get("SystemRoot") or r"C:\Windows"
    search_dirs = [
        Path(path).resolve().parent,
        Path(sys.executable).resolve().parent,
        # PyInstaller puts the bundled dependencies next to the executable, in `_internal`.
        Path(getattr(sys, "_MEIPASS", sys.executable)).resolve(),
        Path(sys.base_prefix),
        Path(sys.base_prefix) / "DLLs",
        Path(sys.prefix),
        Path(sys.prefix) / "Library" / "bin",
        Path(root) / "System32",
        Path(root) / "SysWOW64",
    ]
    search_dirs.extend(Path(entry) for entry in str(env.get("PATH", "")).split(os.pathsep) if entry)
    missing = []
    for name in image.imports:
        lowered = name.lower()
        if lowered.startswith(_API_SET_PREFIXES):
            continue
        if any((directory / name).is_file() for directory in search_dirs):
            continue
        missing.append(name)
    return tuple(missing)


@dataclass(frozen=True)
class MeterWheelDllReport:
    """Operator-facing answer to "why can the meter-wheel DLL not be used on this machine?"."""

    searched: tuple[str, ...]
    found_path: str = ""
    load_error: str = ""
    machine: str = ""
    expected_machine: str = "x64"
    imports: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    driver_hint: str = ""

    @property
    def loadable(self) -> bool:
        return bool(self.found_path) and not self.load_error and not self._bitness_mismatch

    @property
    def _bitness_mismatch(self) -> bool:
        return bool(self.machine) and self.machine != self.expected_machine

    def lines(self) -> tuple[str, ...]:
        lines = ["LSI-8181 米輪 DLL 診斷"]
        lines.append("  搜尋順序：" + " → ".join(self.searched))
        if not self.found_path:
            lines.append(
                "  [FAIL] 找不到 DLL；請以 CCD 頁的「瀏覽」指定 LSI8181_64.dll、"
                "把 DLL 放在 EXE 同一資料夾，或安裝驅動。"
                "（打包版不會用檔名搜尋系統路徑，必須給完整路徑或放在 EXE 旁）"
            )
            return tuple(lines)
        lines.append(f"  [ OK ] 檔案：{self.found_path}")
        if self.machine:
            status = " OK " if not self._bitness_mismatch else "FAIL"
            lines.append(f"  [{status}] 位元數：{self.machine}（需要 {self.expected_machine}）")
        if self.load_error:
            lines.append(f"  [FAIL] 載入：{self.load_error}")
        if self.imports:
            lines.append(f"  匯入 {len(self.imports)} 個 DLL：{'、'.join(self.imports[:8])}"
                         + ("…" if len(self.imports) > 8 else ""))
        if self.missing:
            if self.load_error:
                lines.append(f"  [FAIL] 缺少相依 DLL（最可能的原因）：{'、'.join(self.missing)}")
            else:
                lines.append(
                    f"  [注意] 下列相依無法從搜尋路徑解析（DLL 本身可載入，可能是以完整路徑載入的私有 DLL）："
                    f"{'、'.join(self.missing)}"
                )
        elif self.load_error and "126" in self.load_error:
            lines.append("  [注意] 相依清單都在，126 可能來自延遲載入或驅動服務未啟動。")
        if self.driver_hint:
            lines.append(f"  [注意] {self.driver_hint}")
        return tuple(lines)

    def summary(self) -> str:
        """One copyable line for the field."""

        if not self.found_path:
            return "LSI DLL 找不到（見搜尋順序）"
        if self._bitness_mismatch:
            return f"LSI DLL 位元數不符（{self.machine}）"
        if self.load_error and self.missing:
            return "LSI DLL 缺相依：" + "、".join(self.missing[:4])
        if self.load_error:
            return f"LSI DLL 載入失敗：{self.load_error[:60]}"
        return "LSI DLL 可載入"


def diagnose_meter_wheel_dll(
    dll_path: str | os.PathLike | None = None,
    environ: Mapping[str, str] | None = None,
) -> MeterWheelDllReport:
    """Inspect the meter-wheel DLL without opening it: search paths, bitness, imports, load error."""

    from devices.lsi8181 import Lsi8181Library, Lsi8181LoadError  # noqa: PLC0415 - avoid a cycle at import time

    env = os.environ if environ is None else environ
    searched = dll_candidates(dll_path=dll_path, environ=env)
    found = next((candidate for candidate in searched if Path(candidate).is_file()), "")
    # An explicit path or an env var is reported verbatim even when the file is absent.
    if not found and dll_path is not None:
        found = str(dll_path)
    if not found:
        return MeterWheelDllReport(searched=searched)

    image = read_pe_image(found)
    load_error = ""
    driver_hint = ""
    try:
        Lsi8181Library.load(dll_path=found, environ=env)
    except Lsi8181LoadError as exc:
        load_error = str(exc)
        if env.get(DLL_PATH_ENV) and str(env[DLL_PATH_ENV]) != found:
            driver_hint = f"環境變數 {DLL_PATH_ENV} 指向 {env[DLL_PATH_ENV]}，可能不是實際要用的 DLL。"
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        load_error = f"{type(exc).__name__}: {exc}"

    return MeterWheelDllReport(
        searched=searched,
        found_path=found,
        load_error="" if not load_error else load_error,
        machine=image.machine if image else "",
        imports=image.imports if image else (),
        missing=missing_dependencies(found, image, env) if image else (),
        driver_hint=driver_hint,
    )


__all__ = [
    "DLL_NAME",
    "MeterWheelDllReport",
    "PeImage",
    "diagnose_meter_wheel_dll",
    "missing_dependencies",
    "read_pe_image",
]
