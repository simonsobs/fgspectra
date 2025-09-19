try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations or if setuptools-scm fails
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version("fgspectra")
        except PackageNotFoundError:
            __version__ = "unknown (not installed)"
    except ImportError:
        # Python < 3.8 fallback
        try:
            import pkg_resources
            __version__ = pkg_resources.get_distribution("fgspectra").version
        except Exception:
            __version__ = "unknown"
