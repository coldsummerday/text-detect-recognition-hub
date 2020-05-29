import sys
from time import time

class ProgressBar(object):
    """A progress bar which can print the progress"""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write('[{}] 0/{}, elapsed: 0s, ETA:'.format(
                ' ' * self.bar_width, self.task_num))
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self):
        self.completed += 1
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s' \
                  ''.format(self.completed, self.task_num, fps,
                            int(elapsed + 0.5), eta)

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                'completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                    self.completed, int(elapsed + 0.5), fps))
        self.file.flush()



class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)

class Timer(object):
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> import mmcv
    >>> with mmcv.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = mmcv.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        """Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur