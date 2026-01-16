def test_import_split_beam():
    import importlib
    # import the utils package and the module
    utils = importlib.import_module('mkp_sci_com.utils')
    assert hasattr(utils, 'split_beam')
    mod = importlib.import_module('mkp_sci_com.utils.split_beam')
    assert hasattr(mod, 'split_beam')
    assert callable(mod.split_beam)
