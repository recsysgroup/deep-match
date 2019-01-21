import tensorflow as tf


class LogviewMetricWriter(object):
    def __init__(self):
        self.writer = tf.summary.MetricsWriter('.', flush_secs=10)

    def add_scalar(self, _metric_values, _step):
        for k, v in _metric_values.items():
            self.writer.add_scalar(k, v, _step)

    def __del__(self):
        self.writer.close()


class LogviewMetricHook(tf.train.SessionRunHook):
    def __init__(self, _metric_ops, _step_op, _logviewMetricWriter):
        print _step_op
        self._step_op = _step_op
        self._metric_ops = {}
        self._logviewMetricWriter = _logviewMetricWriter
        for k, v in _metric_ops.items():
            self._metric_ops[k] = v[0]

    def end(self, session):
        if self._metric_ops is not None:
            _metric_values, _step = session.run([self._metric_ops, self._step_op])
            self._logviewMetricWriter.add_scalar(_metric_values, _step)
