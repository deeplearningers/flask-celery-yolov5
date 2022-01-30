from threading import Thread


class DataThread(Thread):

    def __init__(self,q,func, args=()):
        super(DataThread, self).__init__()
        self.func = func
        self.args = args
        self.q = q

    def run(self):
        self.result = self.func(*self.args)
        self.q.put(self.result)

class PredictThread(Thread):

    def __init__(self,func, args=()):
        super(PredictThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return []
