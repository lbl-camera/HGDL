from loguru import logger

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError) as ex:
    raise RuntimeError('Running hgdl from source code requires installation. If you would like an editable source '
                       'install, use "pip install -e ." to perform and editable installation.') from ex


logger.disable('hgdl')
