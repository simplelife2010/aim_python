"""AIM-specific TensorFlow operations
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.contrib import signal as contrib_signal

def read_audio(file_pattern, name=None):
  """Read next audio file matching filename patterns and return decoded audio
    using an iterator.
  
  Args:
    file_pattern: A string or scalar string `Tensor`, representing the
    filename pattern that will be matched.
    
    name: `string`, name of the operation.
	
  Returns:
    An initializer operation for the iterator
    A `Tensor` containing the name of the decoded file.
    A `Tensor` containing the decoded audio.
  """
  with tf.name_scope(name, "read_audio"):
    files = tf.data.Dataset.list_files(file_pattern)
    iterator = files.make_initializable_iterator()
    next_file = iterator.get_next(name="next_file")
    read_file = tf.read_file(next_file, name="read_file")
    audio, sample_rate = contrib_audio.decode_wav(read_file, name="decode")
    decoded_audio = tf.reduce_mean(audio, axis=-1, keepdims=False,
      name="mixdown")
    return (iterator.initializer,
            next_file,
            decoded_audio)
            
def read_ground_truth_labels(audio_filename, fr_length, sample_rate, name=None):
  """Read the corresponding ground truth labels file for the given audio file.
    For an audio filename `test.wav` the ground truth filename is expected to be
    `test.gt`.
  
  Args:
    audio_filename: A string or scalar string `Tensor`, containing the audio
    filename.
    fr_length: An integer scalar `Tensor`. The window length in samples.
    sample_rate: Python float. Samples per second of the input signal used to
    create the spectrogram. We need this to figure out the actual frequencies
    for each spectrogram bin, which dictates how they are mapped into the mel
    scale.
      
  Returns:
    A `Tensor` containing the labels.
  """
  with tf.name_scope(name, "read_gt"):
    split = tf.string_split(
      [audio_filename], delimiter=".", skip_empty=False).values
    ext_stripped = split[:-1]
    ext_concatenated = tf.concat([ext_stripped, ["gt"]], 0)
    gt_filename = tf.reduce_join(ext_concatenated, separator=".")
    read_file = tf.read_file(gt_filename)
    # Separate input file by line break ("\r\n")
    rows = tf.string_split([read_file], delimiter="\r\n").values
    # Decode into three columns (float, float, int)
    csv_columns = tf.decode_csv(rows, [[], [], []], field_delim="\t")
    # Get end timestamp of each segment
    segments = csv_columns[1]
    # Get labels
    labels = tf.to_int32(csv_columns[2])
    # Calculate frame length in seconds
    fr_length_s = tf.divide(tf.to_float(fr_length), sample_rate)
    # Convert segment timestamps into frame indices
    frame_indices = tf.to_int32(tf.divide(segments, fr_length_s))
    # Magic 1: Expand by getting a row of ones until the frame index
    sm = tf.sequence_mask(
      frame_indices, tf.reduce_max(frame_indices), dtype=tf.int32)
    # Magic 2: Subtract each row from the next row
    diff = tf.concat([[sm[0,:]], tf.subtract(sm[1:,:], sm[:-1,:])], 0)
    # Magic 3: Reduce the matrix on the other axis to get labels by frame
    labels_by_frame = tf.reduce_max(
      tf.multiply(labels, tf.transpose(diff)), axis=-1)
  return labels_by_frame

def remove_dc(frames, name=None):
  """Remove dc offset from a batch of audio signals.

  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    name: `string`, name of the operation.
  
  Returns:
    A `Tensor` with the same shape as the input.
  """
  with tf.name_scope(name, "remove_dc"):
    dc_offset = tf.reduce_mean(frames, axis=-1, keepdims=True, name="dc_offset")
    return tf.subtract(frames, dc_offset, name="remove_dc")
    
def scale(frames, attenuation, name=None):
  """Scale a batch of audio signals using root mean square normalization.
  
  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    attenuation: `float32`, the attenuation in decibel.
    name: `string`, name of the operation.
  
  Returns:
    A `Tensor` with the same shape as the input.
  """
  with tf.name_scope(name, "scale"):
    frames_rms = tf.sqrt(
      tf.reduce_mean(
        tf.square(frames), axis=-1, keepdims=True))
    exponent = tf.divide(attenuation, 20.0)
    attenuation_lin = tf.pow(10.0, exponent)
    normalized_frames = tf.divide(
      frames, tf.multiply(
        frames_rms, attenuation_lin))
    return normalized_frames
  
    
def normalize_audio(frames, attenuation, name=None):
  """Normalize a batch of audio signals by removing dc offset and applying
    rms normalization.
  
  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    attenuation: `float32`, the attenuation in decibel.
    name: `string`, name of the operation.
    
  Returns:
    A `Tensor` with the same shape as the input.
  """
  with tf.name_scope(name, "norm_audio"):
    frames_ac = remove_dc(frames)
    return scale(frames_ac, attenuation)
    
def mag_spectrogram(frames, fft_length=1024, fft_step=512, name=None):
  """Extract magnitude spectrograms from a batch of audio signals.
  
  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    fft_length: An integer scalar `Tensor`. The window length in samples.
    fft_step: An integer scalar `Tensor`. The number of samples to step.
    name: `string`, name of the operation.
    
  Returns:
    A `Tensor` with shape `[frames, spectrogram_bins]`.
  """
  with tf.name_scope(name, "mag_spectrogram"):
    stft = contrib_signal.stft(frames, fft_length, fft_step)
    ms = tf.abs(stft)
    return ms
    
def log_mel_spectrogram(
  mag_spectrogram, sample_rate, lower_edge_hertz, upper_edge_hertz,
  n_mel_bins=128, name=None):
  """Calculate log mel spectrograms from a batch of magnitude spectrograms.
  
  Args:
    mag_spectrogram: A `Tensor` of shape `[frames, spectrogram_bins]`.
    n_mel_bins: An integer scalar `Tensor`. The number of mel bins to
    calculate.
    sample_rate: Python float. Samples per second of the input signal used to
    create the spectrogram. We need this to figure out the actual frequencies
    for each spectrogram bin, which dictates how they are mapped into the mel
    scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
    included in the mel spectrum. This corresponds to the lower edge of the
    lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
    frequency band.
    name: `string`, name of the operation.
  
  Returns:
    A `Tensor` with shape `[frames, mel_bins]`.
  """
  with tf.name_scope(name, "mel_spectrogram"):
    n_spectrogram_bins = mag_spectrogram.shape[-1].value
    mel_matrix = contrib_signal.linear_to_mel_weight_matrix(
      n_mel_bins, n_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrogram = tf.tensordot(mag_spectrogram, mel_matrix, 1)
    mel_spectrogram.set_shape(mag_spectrogram.shape[:-1].concatenate(
      mel_matrix.shape[-1:]))
    log_offset = 1e-6
    log_mel_spectrogram = tf.log(mel_spectrogram + log_offset)
    return log_mel_spectrogram

def mfcc(log_mel_spectrogram, n_mfcc=20, name=None):
  """Calculate mel frequency cepstral coefficients from a batch of log mel
    spectrograms.
  
  Args:
    log_mel_spectrogram: A `Tensor` of shape `[frames, mel_bins]`.
    n_mfcc: The number of coefficients to extract.
    
  Returns:
    A `Tensor` with shape `[frames, n_mfcc]` containing the mfccs.
    A `Tensor` with shape `[n_mfcc]` containing the feature ids.
  """
  with tf.name_scope(name, "mfcc"):
    indices = tf.range(n_mfcc)
    feature_ids = tf.add("mfcc_", tf.as_string(indices))
    return (feature_ids, contrib_signal.mfccs_from_log_mel_spectrograms(
      log_mel_spectrogram)[..., 0:n_mfcc])
    
def spectral_centroid_bandwidth(mag_spectrogram, sample_rate, p, name=None):
  """Calculate the logarithmized spectral centroid from a batch of magnitude
    spectrograms.
  
  Args:
    mag_spectrogram: A `Tensor` of shape `[frames, spectrogram_bins]`.
    sample_rate: Python float. Samples per second of the input signal.
    p: Power to raise deviation from spectral centroid.
    name: `string`, name of the operation.
    
  Returns:
    A `Tensor` with shape `[frames, 1]` containing the spectral centroid.
    A `Tensor` with shape `[1]` containing the feature id.
  """
  with tf.name_scope(name, "spectral_centroid_bandwidth"):
    eps = 0.00000001
    n_spectrogram_bins = mag_spectrogram.shape[-1].value
    center_frequencies = tf.lin_space(
      0.0, sample_rate / 2.0, int(n_spectrogram_bins))
    length = tf.reduce_sum(mag_spectrogram, axis=-1, keepdims=True)
    normalized_mag_spectrogram = tf.divide(mag_spectrogram, length + eps)
    weighted_spectrogram = tf.multiply(
      center_frequencies, normalized_mag_spectrogram)
    centroid = tf.reduce_sum(weighted_spectrogram, axis=-1, keepdims=True)
    deviation = tf.abs(
      tf.subtract(center_frequencies, centroid))
    bandwidth = tf.pow(tf.reduce_sum(tf.multiply(mag_spectrogram, tf.pow(
      deviation, p)), axis=-1, keepdims=True), tf.reciprocal(p))
    feature_ids = tf.constant(["spectral_centroid", "spectral_bandwidth"])
    return (feature_ids, tf.log(tf.concat([centroid, bandwidth], axis=-1)))

def zero_crossing_rate(frames, sfr_length=1024, sfr_step=512, name=None):
  """Calculate the zero crossing rate for a batch of audio signals.
  
  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    sfr_length: An integer scalar `Tensor`. The subframe length in samples.
    sfr_step: An integer scalar `Tensor`. The number of samples to step.
    name: `string`, name of the operation.
    
  Returns:
    A `Tensor` with shape `[frames, 1]` containing the zero crossing rate.
    A `Tensor` with shape `[1]` containing the feature id.
  """
  with tf.name_scope(name, "zero_crossing_rate"):
    subframes = contrib_signal.frame(frames, sfr_length, sfr_step)
    sign = tf.sign(subframes)
    diff = tf.divide(tf.abs(tf.subtract(sign[:,:,1:], sign[:,:,:-1])), 2.0)
    n_zc = tf.reduce_sum(diff, axis=-1, keepdims=True)
    zcr = tf.divide(n_zc, tf.to_float(sfr_length))
    return ([tf.constant("zero_crossing_rate")], zcr)
    
def extract_features(frames,
                     sample_rate,
                     lower_edge_hertz,
                     upper_edge_hertz,
                     sfr_length=1024,
                     sfr_step=512,
                     n_mel_bins=128,
                     n_mfcc=20,
                     p=2.0,
                     name=None):
  """Extract a batch of features from a batch of audio signals.
  
  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    sample_rate: Python float. Samples per second of the input signal used to
    create the spectrogram. We need this to figure out the actual frequencies
    for each spectrogram bin, which dictates how they are mapped into the mel
    scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
    included in the mel spectrum. This corresponds to the lower edge of the
    lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
    frequency band.
    sfr_length: An integer scalar `Tensor`. The subframe length in samples.
    sfr_step: An integer scalar `Tensor`. The number of samples to step.
    n_mel_bins: An integer scalar `Tensor`. The number of mel bins to
    calculate.
    n_mfcc: An integer scalar `Tensor`. The number of mfcc to calculate.
    p: Power to raise deviation from spectral centroid.
    name: `string`, name of the operation.
  
  Returns:
    A `Tensor` containing the features.
    A `Tensor` containing the feature ids.
  """
  with tf.name_scope(name, "extract_features"):
    mag_s = mag_spectrogram(frames, sfr_length, sfr_step)
    mel_s = log_mel_spectrogram(
      mag_s, sample_rate, lower_edge_hertz, upper_edge_hertz, n_mel_bins)
    mfcc_ids, mfccs = mfcc(mel_s, n_mfcc)
    scb_id, scb = spectral_centroid_bandwidth(mag_s, sample_rate, p)
    zcr_id, zcr = zero_crossing_rate(frames, sfr_length, sfr_step)
    features = tf.concat([mfccs, scb, zcr], axis=-1)
    feature_ids = tf.concat([mfcc_ids, scb_id, zcr_id], axis=0)
    return (features, feature_ids)
    
def reduce_features(features, feature_ids, mean=True, variance=True, name=None):
  """Reduce a batch of features along the time axis for each frame, calculating
    mean, variance etc.
  
  Args:
    features: A `Tensor` of shape `[frames, subframes, features]`.
    feature_ids: A `Tensor` containing the feature ids.
    mean: `bool`, if `true`, calculate the mean.
    variance: `bool`, if `true`, calculate the variance.
    name: `string`, name of the operation.
    
  Returns:
    A `Tensor` of shape `[frames, reduce operations * features]`.
    A `Tensor` containing the reduced feature ids.
  """
  with tf.name_scope(name, "reduce_features"):
    mean = tf.convert_to_tensor(mean, dtype=tf.bool)
    variance = tf.convert_to_tensor(variance, dtype=tf.bool)
    n_frames = tf.shape(features)[0]
    moments = tf.nn.moments(features, axes=[1])
    features_mean = tf.cond(mean,
                            lambda: moments[0],
                            lambda: tf.zeros([n_frames, 0]),
                            name="calculate_mean")
    features_variance = tf.cond(variance,
                                lambda: moments[1],
                                lambda: tf.zeros([n_frames, 0]),
                                name="calculate_variance")
    ids_mean = tf.cond(mean,
                       lambda: tf.add(feature_ids, [tf.constant("_mean")]),
                       lambda: tf.as_string(tf.zeros([0])),
                       name="mean_feature_ids")
    ids_variance = tf.cond(variance,
                           lambda: tf.add(feature_ids, [tf.constant("_variance")]),
                           lambda: tf.as_string(tf.zeros([0])),
                           name="variance_feature_ids")
    reduced_features = tf.concat([features_mean, features_variance], axis=-1)
    reduced_feature_ids = tf.concat([ids_mean, ids_variance], axis=0)
    return (reduced_features, reduced_feature_ids)
    
def normalize_features(features, name=None):
  """Normalize a batch of features to zero mean, unit variance.
  
  Args:
    features: A `Tensor` of shape `[frames, features]`.
    name: `string`, name of the operation.
    
  Returns:
    A `Tensor` of shape `[frames, features]`.
  """
  with tf.name_scope(name, "norm_features"):
    batch_mean, batch_variance = tf.nn.moments(
      features, axes=[0], keep_dims=True)
    normalized_features = tf.nn.batch_normalization(features,
                                                    batch_mean,
                                                    batch_variance,
                                                    offset=None,
                                                    scale=None,
                                                    variance_epsilon=1e-3)
  return normalized_features