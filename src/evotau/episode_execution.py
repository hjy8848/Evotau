"""Console pause signal shared by the serial τ-bench runner."""


class StopBeforeEpisodeDispatch(RuntimeError):
    """Stop cleanly before dispatching the next native τ-bench episode."""
