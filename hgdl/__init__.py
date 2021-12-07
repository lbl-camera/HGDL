#from ._version import get_versions

#__version__ = get_versions()["version"]
#del get_versions

#global_config = None

from . import _version
__version__ = _version.get_versions()['version']
