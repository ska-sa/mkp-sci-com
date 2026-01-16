def test_import_mkpsim():
    import importlib
    mkp = importlib.import_module('mkp_sci_com')
    assert hasattr(mkp, 'mkpsim')
    mkpsim = mkp.mkpsim
    # ensure the sim package can be imported and has expected attributes
    assert mkpsim.__name__ == 'mkp_sci_com.sim'
