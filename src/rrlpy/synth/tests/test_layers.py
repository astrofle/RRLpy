from astropy import units as u


class TestLayers:
    def setup_method(self):
        self.ne = [1] * u.cm**-3
        self.te = [1000] * u.K
        self.l = [1] * u.pc
        self.v_rms = [1] * u.km / u.s

    def test_layers(self):
        pass
