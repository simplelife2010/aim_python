from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import aim_ops as aim
import numpy as np
from sklearn import feature_selection as fs
from tensorflow.contrib import signal as contrib_signal

def main(_):
  with tf.Graph().as_default():
    n_mel_bins = 128
    lower_edge_hertz = 0.0
    upper_edge_hertz = 11025.0
    sample_rate = 22050.0
  
    sfr_length = tf.constant(1024, name="sfr_length")
    sfr_step = tf.constant(512, name="sfr_step")
  
    file_pattern = tf.placeholder(tf.string, shape=(), name="file_pattern")
    fr_length = tf.placeholder(tf.int32, shape=(), name="fr_length")
    fr_step = tf.placeholder(tf.int32, shape=(), name="fr_step")
    attenuation = tf.placeholder(tf.float32, shape=(), name="attn")
    n_mfcc = tf.placeholder(tf.int32, shape=(), name="n_mfcc")
    calculate_mean = tf.placeholder(tf.bool, shape=(), name="cal_mean")
    calculate_variance = tf.placeholder(
      tf.bool, shape=(), name="cal_variance")
    p_deviation = tf.placeholder(tf.float32, shape=(), name="p_deviation")
    
    initializer, fn, dc = aim.read_audio(file_pattern)
    gt = aim.read_ground_truth_labels(fn, fr_length, sample_rate)
    frames = contrib_signal.frame(dc, fr_length, fr_step, name="frame_audio")
    nf = aim.normalize_audio(frames, attenuation)
    features, ids = aim.extract_features(nf,
                                    sample_rate,
                                    lower_edge_hertz,
                                    upper_edge_hertz,
                                    sfr_length,
                                    sfr_step,
                                    n_mel_bins,
                                    n_mfcc,
                                    p_deviation)
    rf, rids = aim.reduce_features(
      features, ids, calculate_mean, calculate_variance)
    nf = aim.normalize_features(rf)
    with tf.Session() as sess:
      file_writer = tf.summary.FileWriter('log', sess.graph)
      parameters = {
        file_pattern: "*.wav",
        fr_length: 11025,
        fr_step: 11025,
        attenuation: 24.0,
        n_mfcc: 20,
        calculate_mean: True,
        calculate_variance: True,
        p_deviation: 2.0
      }
      sess.run(initializer, parameters)
      while True:
        try:
          rid, f, n, g = sess.run([rids, fn, nf, gt], parameters)
          print(f.decode())
          mi = fs.mutual_info_classif(n, g)
          mi_with_ids = np.transpose(np.concatenate(([rid], [mi]), axis=0))
          print(mi_with_ids)
        except tf.errors.OutOfRangeError:
          break

          
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)