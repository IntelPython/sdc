import sys
import builtins


def _init_extension():
    '''Register Pandas classes and functions with Numba.

    This entry_point is called by Numba when it initializes.
    '''
    if 'pandas' in sys.modules:
        import sdc
    else:
        old_import = builtins.__import__
        def new_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'pandas':
                current_import = builtins.__import__
                builtins.__import__ = old_import
                import pandas
                import sdc
                builtins.__import__ = current_import
            return old_import(name, globals, locals, fromlist, level)
        builtins.__import__ = new_import

        # class Finder:
        #     def find_module(self, fullname, path=None):
        #         if fullname.startswith('pandas'):
        #             import sdc
        # sys.meta_path.insert(0, Finder())
