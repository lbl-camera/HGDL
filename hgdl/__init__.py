from loguru import logger

from . import _version

__version__ = _version.get_versions()['version']

logger.disable('hgdl')
