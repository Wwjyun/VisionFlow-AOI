"""Tests for the meter-wheel DLL diagnostics (`devices/meter_wheel_dll.py`).

The camera machine reported only "米輪的 DLL 不能用"; Windows gives the same text for a missing DLL and
for a missing dependency, so these tests pin the parts that let the field report the real cause:
search order, PE bitness, import listing and the load error text.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from devices.lsi8181 import DLL_NAME, DLL_PATH_ENV, Lsi8181LoadError, dll_candidates
from devices.meter_wheel_dll import (
    diagnose_meter_wheel_dll,
    missing_dependencies,
    read_pe_image,
)

X64_PYD = Path(sys.base_prefix) / "DLLs" / "_ssl.pyd"
X86_DLL = Path(sys.prefix) / "Lib" / "site-packages" / "clr_loader" / "ffi" / "dlls" / "x86" / "ClrLoader.dll"


class SearchOrderTests(unittest.TestCase):
    def test_environment_variable_wins_and_is_the_only_candidate(self):
        self.assertEqual(dll_candidates(environ={DLL_PATH_ENV: r"C:\vendor\LSI8181_64.dll"}),
                         (r"C:\vendor\LSI8181_64.dll",))

    def test_without_an_override_the_bare_name_is_always_tried(self):
        candidates = dll_candidates(environ={})
        self.assertEqual(candidates[-1], DLL_NAME, "the Windows search path must stay the last resort")

    def test_an_existing_copy_next_to_the_executable_is_tried_first(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_lsi_") as directory:
            expected = Path(directory) / DLL_NAME
            expected.write_bytes(b"MZ")
            candidates = dll_candidates(environ={}, dll_path=expected)
        self.assertEqual(candidates, (str(expected),))


class PeInspectionTests(unittest.TestCase):
    def test_a_real_extension_module_is_read_as_x64_with_its_imports(self):
        if not X64_PYD.is_file():
            self.skipTest(f"missing fixture: {X64_PYD}")
        image = read_pe_image(X64_PYD)
        self.assertIsNotNone(image)
        self.assertEqual(image.machine, "x64")
        self.assertTrue(image.imports, "a compiled extension module must import something")
        self.assertEqual(missing_dependencies(X64_PYD, image), (),
                         "the interpreter's own DLLs must resolve on any development machine")

    def test_a_non_pe_file_is_reported_as_unreadable_instead_of_raising(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_lsi_") as directory:
            text_file = Path(directory) / "LSI8181_64.dll"
            text_file.write_text("this is not a PE image", encoding="utf-8")
            self.assertIsNone(read_pe_image(text_file))

    def test_a_missing_file_is_reported_as_unreadable(self):
        self.assertIsNone(read_pe_image(Path(tempfile.gettempdir()) / "definitely_absent_visionflow.dll"))

    def test_a_32_bit_dll_is_named_as_a_bitness_mismatch(self):
        if not X86_DLL.is_file():
            self.skipTest(f"missing fixture: {X86_DLL}")
        report = diagnose_meter_wheel_dll(dll_path=X86_DLL, environ={})
        self.assertEqual(report.machine, "x86")
        self.assertFalse(report.loadable)
        self.assertIn("位元數", report.summary())
        self.assertIn("x86", "\n".join(report.lines()))


class DiagnosisTests(unittest.TestCase):
    def test_a_missing_dll_reports_the_search_order(self):
        report = diagnose_meter_wheel_dll(environ={DLL_PATH_ENV: r"C:\absent\LSI8181_64.dll"})
        self.assertFalse(report.loadable)
        self.assertIn("找不到", report.summary())
        text = "\n".join(report.lines())
        self.assertIn(DLL_NAME, text)
        self.assertIn("搜尋順序", text)

    def test_the_dll_search_order_is_the_loader_search_order(self):
        environ = {DLL_PATH_ENV: r"C:\vendor\LSI8181_64.dll"}
        report = diagnose_meter_wheel_dll(environ=environ)
        self.assertEqual(report.searched, dll_candidates(environ=environ))

    def test_a_file_that_is_not_a_dll_reports_a_load_error_without_raising(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_lsi_") as directory:
            fake = Path(directory) / DLL_NAME
            fake.write_text("not a dll", encoding="utf-8")
            report = diagnose_meter_wheel_dll(dll_path=fake, environ={})
        self.assertFalse(report.loadable)
        self.assertTrue(report.load_error)
        self.assertIn("LSI DLL", report.summary())

    def test_the_report_lines_name_every_fact_the_field_needs(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_lsi_") as directory:
            fake = Path(directory) / DLL_NAME
            fake.write_text("not a dll", encoding="utf-8")
            lines = diagnose_meter_wheel_dll(dll_path=fake, environ={}).lines()
        joined = "\n".join(lines)
        for expected in ("LSI-8181 米輪 DLL 診斷", "搜尋順序", "檔案：", "載入："):
            with self.subTest(expected=expected):
                self.assertIn(expected, joined)

    def test_an_environment_override_that_differs_from_the_used_file_is_flagged(self):
        if not X86_DLL.is_file():
            self.skipTest(f"missing fixture: {X86_DLL}")
        report = diagnose_meter_wheel_dll(
            dll_path=X86_DLL, environ={DLL_PATH_ENV: r"C:\other\LSI8181_64.dll"}
        )
        self.assertIn(DLL_PATH_ENV, report.driver_hint)


class LoaderContractTests(unittest.TestCase):
    def test_the_loader_rejects_an_explicit_path_that_does_not_exist(self):
        with self.assertRaises(Lsi8181LoadError) as caught:
            from devices.lsi8181 import Lsi8181Library

            Lsi8181Library.load(dll_path=r"C:\absent\LSI8181_64.dll", environ={})
        self.assertIn("找不到", str(caught.exception))

    def test_a_windows_error_code_is_named_in_the_load_failure(self):
        from devices.lsi8181 import _windows_error_text

        error = OSError("boom")
        error.winerror = 126
        self.assertIn("126", _windows_error_text(error))
        self.assertIn("相依", _windows_error_text(error))
        error.winerror = 193
        self.assertIn("64 位元", _windows_error_text(error))


if __name__ == "__main__":
    unittest.main()
