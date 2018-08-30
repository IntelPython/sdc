import ctypes
import os

def bind(ll, dll_name, symbols, path=None):
    '''
    Bind symbols with the same name to llvm.
    Return ctypes CDLL for given lib

    Params:
        ll llvmlite.binding
        dll_name filename of library
        symbols list of symbols, bind all if empty, no binding if None
        path optional path to find library, HPAT root by default
    '''
    if path == None:
        path = os.path.dirname(os.path.dirname(globals()['__file__']))
    dll = ctypes.cdll.LoadLibrary(os.path.join(path, dll_name))
    if symbols != None:
        if False and len(symbols): # FIXME
            for x in symbols:
                ll.add_symbol(x, ctypes.addressof(getattr(dll, x)))
        else:
            ll.load_library_permanently(os.path.join(path, dll_name))
    return dll
