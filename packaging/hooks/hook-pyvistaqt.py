"""PyInstaller hook for pyvistaqt and VTK modules.

PyVista dynamically imports many VTK modules at runtime.
This hook ensures they are all collected.
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('pyvistaqt')
hiddenimports += collect_submodules('vtkmodules')
hiddenimports += ['vtkmodules.all']
