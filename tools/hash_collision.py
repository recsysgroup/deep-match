import tensorflow as tf
import argparse
import sys
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

# pai
flags.DEFINE_integer("task_index", 0, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("worker_count", 1, "")

# common
flags.DEFINE_string("table", None, "")
flags.DEFINE_string("field", None, "")
flags.DEFINE_integer("size", None, "")


# task



def input_fn_builder():
    _recode_defaults = ['']

    def input_fn():
        d = tf.data.TableRecordDataset([FLAGS.table],
                                       record_defaults=_recode_defaults,
                                       selected_cols=FLAGS.field,
                                       slice_id=0,
                                       slice_count=1)

        d = d.repeat(1)
        d = d.shuffle(buffer_size=1000)
        d = d.batch(batch_size=1000)

        return d

    return input_fn


def stat(id_2_cnt, id_2_hash):
    id_num = 0
    all_times = 0
    id_collision_num = 0
    id_collision_times = 0

    hash_2_ids = {}
    for _id, _hash in id_2_hash.items():
        if _hash not in hash_2_ids:
            hash_2_ids[_hash] = []
        hash_2_ids[_hash].append(_id)

    for _hash, _ids in hash_2_ids.items():

        id_num += len(_ids)
        for _id in _ids:
            all_times += id_2_cnt.get(_id)

        if len(_ids) > 1:
            id_collision_num += len(_ids)
            for _id in _ids:
                id_collision_times += id_2_cnt.get(_id)

    print('id num is {0}'.format(id_num))
    print('id collision num is {0}'.format(id_collision_num))
    print('all times is {0}'.format(all_times))
    print('id collision times is {0}'.format(id_collision_times))
    print('id collision rate is {0}'.format(id_collision_num / (id_num + 0.0)))
    print('times collision rate is {0}'.format(id_collision_times / (all_times + 0.0)))


def main(_):
    print(FLAGS.task_index, FLAGS.job_name, FLAGS.worker_count)

    train_input_fn = input_fn_builder()

    iterator = train_input_fn().make_one_shot_iterator()
    id = tf.cast(iterator.get_next()[0], dtype=tf.string)
    hash = tf.string_to_hash_bucket_fast(id, FLAGS.size)
    id_2_cnt = {}
    id_2_hash = {}

    with tf.Session() as sess:
        try:
            while True:
                _id, _hash = sess.run([id, hash])
                for index in range(len(_id)):
                    id_2_cnt[_id[index]] = id_2_cnt.get(_id[index], 0) + 1
                    id_2_hash[_id[index]] = _hash[index]
        except tf.errors.OutOfRangeError:
            pass

    stat(id_2_cnt, id_2_hash)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
