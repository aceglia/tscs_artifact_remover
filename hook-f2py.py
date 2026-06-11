import sys
import types

module = types.ModuleType("numpy.f2py")
sys.modules["numpy.f2py"] = module
