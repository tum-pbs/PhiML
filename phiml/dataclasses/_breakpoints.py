import bdb
import inspect
import os
import pdb
import sys
from typing import Dict, List, Tuple


class BreakpointDetector:
    """Detect and manage breakpoints in Python code."""

    @staticmethod
    def get_pdb_breakpoints() -> Dict[str, List[int]]:
        """
        Get all breakpoints set in pdb.

        Returns:
            Dict mapping filename -> list of line numbers with breakpoints
        """
        breakpoints = {}
        # Access pdb's internal breakpoint storage
        if hasattr(pdb, 'Pdb') and hasattr(pdb.Pdb, 'get_all_breaks'):
            # For newer Python versions
            try:
                all_breaks = pdb.Pdb().get_all_breaks()
                for filename, linenos in all_breaks.items():
                    breakpoints[filename] = list(linenos)
            except:
                pass
        # Alternative method using bdb module
        try:
            # Check the global breakpoint registry
            if hasattr(bdb.Bdb, 'get_all_breaks'):
                debugger = bdb.Bdb()
                all_breaks = debugger.get_all_breaks()
                for filename, linenos in all_breaks.items():
                    breakpoints[filename] = list(linenos)
        except:
            pass
        # Check sys.breakpointhook (Python 3.7+)
        if hasattr(sys, 'breakpointhook') and sys.breakpointhook != sys.__breakpointhook__:
            # Custom breakpoint hook is set, indicating debugging is active
            breakpoints['__custom_hook__'] = [0]
        return breakpoints

    @staticmethod
    def get_pycharm_breakpoints(filename: str, start_line: int, end_line: int) -> List[int]:
        """
        Attempt to detect PyCharm breakpoints by checking pydevd internals.

        Returns:
            List of line numbers that might have breakpoints
        """
        filename = os.path.normcase(os.path.abspath(filename))
        breakpoints = []
        try:
            # Check if pydevd is available
            if 'pydevd' in sys.modules:
                import pydevd
                # Try to access PyCharm's breakpoint manager
                if hasattr(pydevd, 'GetGlobalDebugger'):
                    debugger = pydevd.GetGlobalDebugger()
                    if debugger is not None:
                        # Check if we can access breakpoints
                        if hasattr(debugger, 'breakpoints'):
                            bp_dict = debugger.breakpoints
                            bp_dict = {os.path.normcase(os.path.abspath(k)): v for k, v in bp_dict.items()}
                            if filename in bp_dict:
                                file_breakpoints = bp_dict[filename]
                                for line_no in file_breakpoints:
                                    if start_line <= line_no <= end_line:
                                        breakpoints.append(line_no)

                        # Alternative: check break_on_exception or other indicators
                        if hasattr(debugger, 'break_on_exceptions'):
                            # This indicates debugging is very active
                            pass
            # Also check pydevd_file_utils if available
            if 'pydevd_file_utils' in sys.modules:
                # PyCharm sometimes stores breakpoint info here
                pass
        except Exception:
            # PyCharm's internal API can be unstable, so we catch everything
            pass
        return breakpoints

    @staticmethod
    def has_trace_function() -> bool:
        """Check if a trace function is currently set (indicating debugging)."""
        return sys.gettrace() is not None

    @staticmethod
    def detect_ide_debugger() -> bool:
        """Detect if running under common IDE debuggers."""
        return any(m in sys.modules for m in ['pydevd', 'debugpy', 'ptvsd', 'pydev_ipython', 'IPython.terminal.debugger'])

    @staticmethod
    def get_function_line_range(func) -> Tuple[str, int, int]:
        """Get the file and line range for a function."""
        try:
            filename = inspect.getfile(func)
            source_lines, start_line = inspect.getsourcelines(func)
            end_line = start_line + len(source_lines) - 1
            return filename, start_line, end_line
        except:
            return None, None, None


def detect_breakpoints_in_property(property_obj) -> List[int]:
    """
    Detect breakpoints in a property and return debugging information.

    Args:
        property_obj: A `property` object

    Returns:
        Line numbers on which breakpoints were detected.
    """
    getter_func = property_obj.fget
    detector = BreakpointDetector()
    filename, start_line, end_line = detector.get_function_line_range(getter_func)
    if not filename:
        return []
    breakpoints = []
    # --- Standard Python breakpoints ---
    all_breakpoints = detector.get_pdb_breakpoints()
    if filename in all_breakpoints:
        function_breakpoints = [line for line in all_breakpoints[filename] if start_line <= line <= end_line]
        breakpoints.extend(function_breakpoints)
    # --- PyCharm breakpoints ---
    pycharm_breakpoints = detector.get_pycharm_breakpoints(filename, start_line, end_line)
    breakpoints.extend(pycharm_breakpoints)
    return breakpoints
