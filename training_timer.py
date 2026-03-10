import time
from datetime import datetime, timedelta


class TrainingTimer:
    """Tracks per-epoch training time and estimates completion time.

    Usage:
        timer = TrainingTimer(total_epochs)
        for epoch in range(1, total_epochs + 1):
            timer.start_epoch()
            # ... training code ...
            timer.end_epoch(epoch, total_epochs)
        timer.summary()
    """

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_durations = []
        self._epoch_start = None
        self._training_start = time.time()

    def start_epoch(self):
        """Call at the beginning of each epoch."""
        self._epoch_start = time.time()

    def end_epoch(self, current_epoch, total_epochs=None):
        """Call at the end of each epoch. Prints timing info and ETA.

        Args:
            current_epoch: 1-based index of the epoch that just finished.
            total_epochs: Override total epochs if needed (uses init value otherwise).
        """
        if self._epoch_start is None:
            return
        duration = time.time() - self._epoch_start
        self.epoch_durations.append(duration)
        self._epoch_start = None

        total = total_epochs or self.total_epochs
        remaining = total - current_epoch
        avg_duration = sum(self.epoch_durations) / len(self.epoch_durations)
        eta_seconds = avg_duration * remaining
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        elapsed = time.time() - self._training_start

        print(
            f"  [Timer] Epoch {current_epoch}/{total} took {duration:.1f}s | "
            f"Avg: {avg_duration:.1f}s/epoch | "
            f"Elapsed: {self._format_duration(elapsed)} | "
            f"ETA: {eta_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(~{self._format_duration(eta_seconds)} remaining)"
        )

    def summary(self):
        """Print final training time summary."""
        if not self.epoch_durations:
            print("[Timer] No epochs recorded.")
            return
        total_time = time.time() - self._training_start
        avg = sum(self.epoch_durations) / len(self.epoch_durations)
        fastest = min(self.epoch_durations)
        slowest = max(self.epoch_durations)
        print(
            f"\n[Timer] Training complete in {self._format_duration(total_time)} | "
            f"Avg: {avg:.1f}s | Fastest: {fastest:.1f}s | Slowest: {slowest:.1f}s"
        )

    def get_epoch_durations(self):
        """Return list of per-epoch durations in seconds (for CSV logging)."""
        return list(self.epoch_durations)

    @staticmethod
    def _format_duration(seconds):
        """Format seconds into human-readable HH:MM:SS string."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"
