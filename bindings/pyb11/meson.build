py_mod = import('python')
py = py_mod.find_installation(pure: false)
pyb11_dep = [
  py.dependency(),
  dependency('pybind11')
]

incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

#   # We do need an absolute path to feed to `cc.find_library` below
#   _incdir_numpy_abs = run_command(py3,
#     ['-c', 'import numpy; print(numpy.get_include())'],
#     check: true
#   ).stdout().strip()
# else
#   _incdir_numpy_abs = incdir_numpy
# endif
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

py.extension_module(
  '_xts_linalg',
  sources : [
    'xtsci_linalg_pyb11.cc',
  ],
  dependencies: [pyb11_dep, xtsci_linalg_dep, np_dep],
  cpp_args: _args,
  install: true,
)
