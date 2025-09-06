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
  'saftig/evaluation/dataset.py',
  'saftig/evaluation/metrics.py',
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

python_sources = [
  'saftig/filtering/__init__.py',
  'saftig/filtering/common.py',
  'saftig/filtering/wf.py',
  'saftig/filtering/uwf.py',
  'saftig/filtering/lms.py',
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
