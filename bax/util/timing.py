"""
Timing utilities.
"""

import time
import datetime


class Timer(object):
    """
    Timer class. Thanks to Eli Bendersky, Josiah Yoder, Jonas Adler, Can Kavaklıoğlu,
    and others from https://stackoverflow.com/a/50957722.
    """
    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.2f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '*[TIME] [%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename,'a') as file:
                print(str(datetime.datetime.now())+": ",message,file=file)
