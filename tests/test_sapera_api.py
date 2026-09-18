"""Unit tests for the pure Sapera API-layer helpers in `devices/sapera_api.py`.

These cover the parts that decide what the operator sees before any hardware exists: where the
machine's managed DLL is looked up, whether the managed DLL and the native runtime versions agree,
how a .NET exception becomes a copyable short code, and the guarantee that a machine without Sapera
LT never loads pythonnet or the .NET runtime.

The real pythonnet side (assembly load, manifest reflection, interop overload selection) is covered
separately by `tests/test_sapera_stub_interop.py`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devices.sapera_api import (
    ACQ_CAPABILITIES,
    ACQ_EVENTS,
    ACQ_PARAMETERS,
    ACQ_VALUES,
    ASSEMBLY_FILE_NAME,
    DEFAULT_SAPERA_DIR,
    DLL_PATH_ENV,
    ERROR_MESSAGES,
    SAPERA_API_MANIFEST,
    SAPERA_NAMESPACE,
    SAPERADIR_ENV,
    AssemblySearch,
    SaperaError,
    SaperaVersions,
    dotnet_exception_name,
    locate_assembly,
    translate_exception,
)


class FakeDotNetException(Exception):
    """Mimics a pythonnet-wrapped .NET exception exposing GetType().FullName."""

    def __init__(self, full_name: str, message: str = "boom"):
        super().__init__(message)
        self._full_name = full_name

    def GetType(self):  # noqa: N802 - .NET naming
        return type("T", (), {"FullName": self._full_name})()


class SaperaErrorTests(unittest.TestCase):
    def test_message_carries_the_code_the_summary_and_the_detail(self):
        error = SaperaError("E-0602", "ExposureTime 寫入失敗")
        self.assertEqual(error.code, "E-0602")
        self.assertEqual(error.detail, "ExposureTime 寫入失敗")
        self.assertIn("E-0602", str(error))
        self.assertIn(ERROR_MESSAGES["E-0602"], str(error))
        self.assertIn("ExposureTime 寫入失敗", str(error))

    def test_message_without_detail_omits_the_separator(self):
        self.assertEqual(str(SaperaError("E-0701")), f"E-0701 {ERROR_MESSAGES['E-0701']}")

    def test_unknown_code_still_produces_a_readable_message(self):
        self.assertIn("E-9999", str(SaperaError("E-9999", "x")))

    def test_every_code_documented_for_the_operator_has_a_summary(self):
        self.assertTrue(ERROR_MESSAGES)
        for code, summary in ERROR_MESSAGES.items():
            self.assertRegex(code, r"^E-\d{4}$")
            self.assertTrue(summary.strip())


class ExceptionTranslationTests(unittest.TestCase):
    def test_sapera_error_passes_through_unchanged(self):
        original = SaperaError("E-0502", "建立失敗")
        self.assertIs(translate_exception(original), original)

    def test_version_mismatch_exceptions_map_to_e0203(self):
        for name in (
            "System.IO.FileLoadException",
            "System.EntryPointNotFoundException",
            "System.MissingMethodException",
            "System.BadImageFormatException",
            "System.TypeLoadException",
            "System.DllNotFoundException",
        ):
            with self.subTest(name=name):
                error = translate_exception(FakeDotNetException(name), "E-0501")
                self.assertEqual(error.code, "E-0203")
                self.assertIn(name, error.detail)

    def test_other_dotnet_failures_keep_the_callers_code(self):
        error = translate_exception(FakeDotNetException("System.InvalidOperationException"), "E-0705")
        self.assertEqual(error.code, "E-0705")
        self.assertIn("System.InvalidOperationException", error.detail)

    def test_default_code_is_the_unexpected_error_code(self):
        self.assertEqual(translate_exception(ValueError("x")).code, "E-0901")

    def test_dotnet_exception_name_falls_back_to_the_python_class(self):
        self.assertEqual(dotnet_exception_name(ValueError("x")), "ValueError")
        self.assertEqual(
            dotnet_exception_name(FakeDotNetException("DALSA.SaperaLT.SapClassBasic.SapException")),
            "DALSA.SaperaLT.SapClassBasic.SapException",
        )

    def test_dotnet_exception_name_survives_a_broken_gettype(self):
        class Broken(Exception):
            def GetType(self):  # noqa: N802 - .NET naming
                raise RuntimeError("no metadata")

        self.assertEqual(dotnet_exception_name(Broken()), "Broken")


class SaperaVersionTests(unittest.TestCase):
    def test_same_major_and_minor_is_not_a_mismatch(self):
        versions = SaperaVersions(assembly_file_version="8.60.0.00.2120", native_file_version="8.60.0.0")
        self.assertFalse(versions.mismatch)
        self.assertIn("8.60", versions.summary())

    def test_the_xx_ccd_failure_case_is_reported_as_a_mismatch(self):
        # xx_ccd was compiled against 9.12 and hit FileLoadException on the 8.6 machine.
        versions = SaperaVersions(assembly_file_version="9.12.0.0", native_file_version="8.60.0.0")
        self.assertTrue(versions.mismatch)
        self.assertIn("9.12", versions.summary())
        self.assertIn("8.60", versions.summary())

    def test_a_missing_or_unparsable_version_never_claims_a_mismatch(self):
        for versions in (
            SaperaVersions(),
            SaperaVersions(assembly_file_version="8.60.0.0"),
            SaperaVersions(native_file_version="8.60.0.0"),
            SaperaVersions(assembly_file_version="unknown", native_file_version="8.60.0.0"),
            SaperaVersions(assembly_file_version="8.60.0.0", native_file_version=""),
        ):
            with self.subTest(versions=versions):
                self.assertFalse(versions.mismatch)

    def test_summary_says_unknown_rather_than_an_empty_string(self):
        self.assertIn("未知", SaperaVersions().summary())


class LocateAssemblyTests(unittest.TestCase):
    def _write_assembly(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        return path

    def test_explicit_environment_path_wins_and_is_not_searched(self):
        with tempfile.TemporaryDirectory() as directory:
            assembly = self._write_assembly(Path(directory) / ASSEMBLY_FILE_NAME)
            search = locate_assembly(environ={DLL_PATH_ENV: str(assembly)})
        self.assertIsInstance(search, AssemblySearch)
        self.assertEqual(search.chosen, str(assembly))
        self.assertIsNone(search.sapera_dir)
        self.assertEqual(search.checked, (str(assembly),))

    def test_explicit_environment_path_that_is_missing_makes_the_camera_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / ASSEMBLY_FILE_NAME
            search = locate_assembly(environ={DLL_PATH_ENV: str(missing)})
        self.assertIsNone(search.chosen)
        self.assertEqual(search.checked, (str(missing),))

    def test_saperadir_tree_is_searched_through_the_known_subpath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self._write_assembly(root / "Components" / "NET" / "Bin" / ASSEMBLY_FILE_NAME)
            search = locate_assembly(environ={SAPERADIR_ENV: str(root)}, default_sapera_dir=str(root / "absent"))
        self.assertEqual(search.sapera_dir, str(root))
        self.assertEqual(search.chosen, str(expected))
        # `checked` records the exact candidate paths tried, so a field report can name them.
        self.assertIn(str(root / "Components" / "NET" / "Bin" / ASSEMBLY_FILE_NAME), search.checked)

    def test_a_demo_copy_is_ranked_below_the_real_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = self._write_assembly(root / "Demos" / "Example" / ASSEMBLY_FILE_NAME)
            real = self._write_assembly(root / "Components" / "NET" / "Bin" / ASSEMBLY_FILE_NAME)
            search = locate_assembly(environ={SAPERADIR_ENV: str(root)}, default_sapera_dir=str(root / "absent"))
        self.assertIn(str(demo), search.found, "the demo copy is still reported as checked")
        self.assertEqual(search.chosen, str(real))

    def test_an_assembly_deeper_than_the_search_limit_is_not_found(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deep = root.joinpath(*[f"level{index}" for index in range(8)], ASSEMBLY_FILE_NAME)
            self._write_assembly(deep)
            search = locate_assembly(environ={SAPERADIR_ENV: str(root)}, default_sapera_dir=str(root / "absent"))
        self.assertIsNone(search.chosen)

    def test_missing_sapera_installation_reports_what_it_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "Sapera"
            search = locate_assembly(environ={}, default_sapera_dir=str(absent))
        self.assertIsNone(search.chosen)
        self.assertIsNone(search.sapera_dir)
        self.assertEqual(search.checked, (str(absent),))

    def test_default_installation_directory_is_used_when_saperadir_is_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self._write_assembly(root / "Components" / "NET" / "Bin" / ASSEMBLY_FILE_NAME)
            search = locate_assembly(environ={}, default_sapera_dir=str(root))
        self.assertEqual(search.chosen, str(expected))


class LoadRuntimeOrderingTests(unittest.TestCase):
    """A machine without Sapera LT must not pay for pythonnet or the .NET runtime."""

    def test_missing_explicit_assembly_fails_before_the_dotnet_runtime_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / ASSEMBLY_FILE_NAME
            with patch("devices.sapera_api.ensure_dotnet_runtime") as bootstrap:
                from devices.sapera_api import load_runtime

                with self.assertRaises(SaperaError) as caught:
                    load_runtime(environ={DLL_PATH_ENV: str(missing)})
        self.assertEqual(caught.exception.code, "E-0201")
        self.assertIn(str(missing), caught.exception.detail)
        bootstrap.assert_not_called()

    def test_no_installation_at_all_fails_with_e0104_before_the_dotnet_runtime_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = str(Path(directory) / "Sapera")
            with patch("devices.sapera_api.ensure_dotnet_runtime") as bootstrap:
                from devices.sapera_api import load_runtime

                with self.assertRaises(SaperaError) as caught:
                    load_runtime(environ={}, default_sapera_dir=absent)
        self.assertEqual(caught.exception.code, "E-0104")
        self.assertIn(absent, caught.exception.detail)
        bootstrap.assert_not_called()


class ManifestTests(unittest.TestCase):
    def test_every_manifest_entry_describes_a_unique_member(self):
        described = [member.describe() for member in SAPERA_API_MANIFEST]
        self.assertEqual(len(described), len(set(described)))

    def test_manifest_covers_every_name_the_camera_writes(self):
        described = " ".join(member.describe() for member in SAPERA_API_MANIFEST)
        for name in ACQ_PARAMETERS + ACQ_VALUES + ACQ_CAPABILITIES:
            with self.subTest(name=name):
                self.assertIn(name, described)

    def test_manifest_types_are_namespaced_unless_they_are_system_types(self):
        from devices.sapera_api import _full_type_name

        self.assertEqual(_full_type_name("SapAcquisition"), f"{SAPERA_NAMESPACE}.SapAcquisition")
        self.assertEqual(_full_type_name("System.Int32"), "System.Int32")

    def test_acquisition_events_include_both_external_trigger_names(self):
        self.assertIn("ExternalTrigger", ACQ_EVENTS)
        self.assertIn("ExternalTrigger2", ACQ_EVENTS)
        self.assertIn("LineTriggerTooFast", ACQ_EVENTS)


if __name__ == "__main__":
    unittest.main()
