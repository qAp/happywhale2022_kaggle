
import importlib


def import_class(s):
    module_name, class_name = s.rsplit(sep='.', maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
