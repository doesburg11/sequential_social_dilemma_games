try:
    from gymnasium.spaces import Discrete
except ImportError:  # pragma: no cover - fallback for legacy gym installs
    from gym.spaces import Discrete


class DiscreteWithDType(Discrete):
    def __init__(self, n, dtype):
        assert n >= 0
        # Gymnasium's Discrete defines `start`; initialize through parent when possible.
        try:
            super().__init__(n=n, start=0, dtype=dtype)
        except TypeError:
            try:
                super().__init__(n=n, dtype=dtype)
            except TypeError:
                # Very old gym fallback without dtype support.
                super().__init__(n=n)
                self.dtype = dtype

        if not hasattr(self, "start"):
            self.start = 0
