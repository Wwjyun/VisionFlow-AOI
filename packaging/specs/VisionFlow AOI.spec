# -*- mode: python ; coding: utf-8 -*-
#
# This spec lives in packaging\specs. PyInstaller resolves relative paths against
# the spec directory, so every source path is derived from the repository root.
from pathlib import Path

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent.parent

cuda_dll = ROOT / 'gpu' / 'visionflow_cuda.dll'
cuda_binaries = [(str(cuda_dll), 'gpu')] if cuda_dll.exists() else []

a = Analysis(
    [str(ROOT / 'gui_launcher.py')],
    pathex=[],
    binaries=cuda_binaries,
    datas=[
        (str(ROOT / 'recipes'), 'recipes'),
        (str(ROOT / 'models' / 'yolox'), 'models/yolox'),
        (str(ROOT / 'build_provenance.json'), '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionFlow AOI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionFlow AOI',
)
