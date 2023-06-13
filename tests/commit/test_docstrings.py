from unittest import TestCase
import importlib


def get_undocumented_wildcards(modulename):
    namespace = importlib.import_module(modulename)
    loc = namespace.__dict__
    undocumented = []
    for key, val in loc.items():
        if (key[0] != "_") and (key not in {"_", "In", "Out", "get_ipython", "exit", "quit", "join", "S", }):
            description = val.__doc__
            if not description:
                import inspect
                if inspect.getdoc(val) is None:
                    undocumented.append(key)
    return undocumented, len(loc.items())


class TestWildcardImportDocs(TestCase):

    def assert_less_undocumented_wc(self, modulename: str, max_undoc_frac: float):
        """
        Assert that the fraction of undocumented public wildcard imports is less than limit.

        Args:
            modulename: Module to be checked
            max_undoc_frac: Limit fraction below which undocumented public wildcard imports are okay.
        """
        undocumented, loc_len = get_undocumented_wildcards(modulename)
        undocumented_fraction = len(undocumented) / loc_len
        self.assertLessEqual(undocumented_fraction, max_undoc_frac, f"{len(undocumented)/loc_len:.2%} of {modulename} imports undocumented. Missing Docstrings in {len(undocumented)}/{loc_len}:\n- " + "\n- ".join(undocumented))

    def test_math(self):
        self.assert_less_undocumented_wc('unifyml.math', 0)

    def test_backend(self):
        self.assert_less_undocumented_wc('unifyml.backend', 0)

    def test_nn(self):
        self.assert_less_undocumented_wc('unifyml.nn', 0)
