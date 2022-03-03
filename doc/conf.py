from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2014-20, Matt Wala and Andreas Kloeckner"

ver_dic = {}
_version_source = "../leap/version.py"
with open(_version_source) as vpy_file:
    version_py = vpy_file.read()

exec(compile(version_py, _version_source, "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]
version = release

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/pymbolic/": None,
    "https://documen.tician.de/dagrt/": None,
    "https://docs.sympy.org/latest/": None,
    }
