import time


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = time.time()#datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = time.time()#datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start)

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
