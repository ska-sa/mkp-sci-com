def test_import_sim():
    import importlib
    mkp = importlib.import_module('mkp_sci_com')
    # sim should be exposed, mkpsim alias should not be present at package level
    assert hasattr(mkp, 'sim')
    assert not hasattr(mkp, 'mkpsim')
    sim = mkp.sim
    # ensure the sim package can be imported and has expected attributes
    assert sim.__name__ == 'mkp_sci_com.sim'
