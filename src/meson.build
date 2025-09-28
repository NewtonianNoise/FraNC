# load meson support for python
py = import('python').find_installation(pure: false)


########################################
# franc
python_sources = [
  'franc/__init__.py',
]

# actually install the python module
py.install_sources(
  python_sources,
  pure: false,
  subdir: 'franc'
)


########################################
# franc.evaluation
python_sources = [
  'franc/evaluation/__init__.py',
  'franc/evaluation/common.py',
  'franc/evaluation/dataset.py',
  'franc/evaluation/metrics.py',
  'franc/evaluation/evaluation.py',
  'franc/evaluation/filter_interface.py',
  'franc/evaluation/report_generation.py',
]

# actually install the python module
py.install_sources(
  python_sources,
  pure: false,
  subdir: 'franc/evaluation'
)


########################################
# franc.signal

python_sources = [
  'franc/filtering/__init__.py',
  'franc/filtering/common.py',
  'franc/filtering/wf.py',
  'franc/filtering/uwf.py',
  'franc/filtering/lms.py',
  'franc/filtering/polylms.py',
]

# actually install the python module
py.install_sources(
  python_sources,
  pure: false,
  subdir: 'franc/filtering'
)


########################################
# franc.evaluation
python_sources_external = [
  'franc/external/__init__.py',
  'franc/external/spicypy_wf.py',
]

# install the franc.external module
py.install_sources(
  python_sources_external,
  pure: false,
  subdir: 'franc/external'
)

########################################
# franc.evaluation
python_sources_common = [
  'franc/common/__init__.py',
  'franc/common/common.py',
]

# install the franc.external module
py.install_sources(
  python_sources_common,
  pure: false,
  subdir: 'franc/common'
)
