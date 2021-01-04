# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['facetrackerNeos.py'],
             pathex=['C:\\facetrackerNeosBuild'],
             binaries=[('dshowcapture/dshowcapture_x86.dll', '.'), ('dshowcapture/dshowcapture_x64.dll', '.'), ('dshowcapture/libminibmcapture32.dll', '.'), ('dshowcapture/libminibmcapture64.dll', '.'), ('escapi/escapi_x86.dll', '.'), ('escapi/escapi_x64.dll', '.'), ('run.bat', '.'), ('msvcp140.dll', '.'), ('vcomp140.dll', '.'), ('concrt140.dll', '.'), ('vccorlib140.dll', '.')],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'mpl-data', 'PyInstaller', 'pywt', 'skimage', 'scipy', 'pyinstaller'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='facetrackerNeos',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='facetrackerNeos')
