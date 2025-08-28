# load meson support for python
py = import('python').find_installation(pure: false)


########################################
# saftig
python_sources = [
  'saftig/__init__.py',
]

# actually install the python module
py.install_sources(
  python_sources,
  pure: false,
  subdir: 'saftig'
)


########################################
# saftig.evaluation
python_sources = [
  'saftig/evaluation/__init__.py',
  'saftig/evaluation/common.py',
  'saftig/evaluation/evaluation.py',
]

# actually install the python module
py.install_sources(
  python_sources,
  pure: false,
  subdir: 'saftig/evaluation'
)


########################################
# saftig.signal
# get numpy dependencies
numpy_dep = dependency('numpy', required: true)

# define build instructions for c extensions
extensions = py.extension_module(
  '_lms_c',
  'saftig/filtering/_lms_c/_lms_c.cpp',
  dependencies: numpy_dep,
  install: true,
  subdir: 'saftig/filtering',
)

python_sources = [
  'saftig/filtering/__init__.py',
  'saftig/filtering/common.py',
  'saftig/filtering/wf.py',
  'saftig/filtering/uwf.py',
  'saftig/filtering/lms.py',
  'saftig/filtering/lms_c.py',
  'saftig/filtering/polylms.py',
]

# actually install the python module
py.install_sources(
  python_sources,
  pure: false,
  subdir: 'saftig/filtering'
)


########################################
# saftig.evaluation
python_sources_external = [
  'saftig/external/__init__.py',
  'saftig/external/spicypy_wf.py',
]

# install the saftig.external module
py.install_sources(
  python_sources_external,
  pure: false,
  subdir: 'saftig/external'
)

########################################
# saftig.evaluation
python_sources_common = [
  'saftig/common/__init__.py',
  'saftig/common/common.py',
]

# install the saftig.external module
py.install_sources(
  python_sources_common,
  pure: false,
  subdir: 'saftig/common'
)
