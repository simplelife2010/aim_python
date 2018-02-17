import sys
import os
import time
import numpy as np
import numpy.random as random
import tensorflow as tf
import scipy.io.wavfile as sw

from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.contrib import signal as contrib_signal
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = 'd:\\Users\\bruno\\aim\\tb'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

def read_audio_op():
	file_pattern = tf.constant('*.wav', name='fn_pattern')
	with tf.name_scope('read_audio') as scope:
		files = tf.data.Dataset.list_files(file_pattern)
		iterator = files.make_one_shot_iterator()
		filename = iterator.get_next(name='next_file')
		wav_audio = tf.read_file(filename, name='read_file')
		decoded_audio = contrib_audio.decode_wav(wav_audio, name='decode')
	return filename, decoded_audio
	
def frame_op(decoded_audio):
	fr_length = tf.placeholder(tf.int32, name='fr_length')
	fr_step = tf.placeholder(tf.int32, name='fr_step')
	with tf.name_scope('frame_audio') as scope:
		frames = contrib_signal.frame(decoded_audio.audio, fr_length, fr_step, axis=0)
		frames_flat = tf.layers.flatten(frames)
	return (fr_length, fr_step, frames_flat)
	
def normalize_audio_op(frames_flat):
	attn_db = tf.constant(24.0, name='attn_24dB')
	with tf.name_scope('norm_audio') as scope:
		with tf.name_scope('remove_dc') as scope:
			dc_offset = tf.reduce_mean(frames_flat, 1, True, name='dc_offset')
			frames_ac = tf.subtract(frames_flat, dc_offset, name='remove_dc')
		with tf.name_scope('norm_rms') as scope:
			with tf.name_scope('rms') as scope:
				frames_s = tf.square(frames_ac)
				frames_ms = tf.reduce_mean(frames_s, 1, True)
				frames_rms = tf.sqrt(frames_ms)
			with tf.name_scope('normalize') as scope:
				attn = tf.pow(10.0, tf.divide(attn_db, 20.0))
				frames_divisor = tf.multiply(frames_rms, attn)
				frames_normalized = tf.divide(frames_ac, frames_divisor)
	return frames_normalized

def features_op(frames_normalized):
	n_mfccs = tf.placeholder(tf.int32, name='num_mfccs')
	with tf.name_scope('extract_features') as scope:
		with tf.name_scope('mfcc') as scope:
			stfts = tf.contrib.signal.stft(frames_normalized,
										   frame_length=1024,
										   frame_step=512)
			magnitude_spectrograms = tf.abs(stfts)
			num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
			
			freq = tf.lin_space(0.0, 22050.0 / 2, int(1 + 1024 // 2))
			
			length = tf.reduce_sum(magnitude_spectrograms, axis=2, keepdims=True)
			normalized_magnitude_spectrograms = tf.divide(magnitude_spectrograms, length)
			weighted_spectrograms = tf.multiply(freq, normalized_magnitude_spectrograms)
			spectral_centroid = tf.multiply(100.0, tf.subtract(tf.log(tf.reduce_sum(weighted_spectrograms, axis=-1)), 7.0))
			
			
			lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 11025.0, 128
			linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
				num_mel_bins, num_spectrogram_bins, 22050, lower_edge_hertz,
				upper_edge_hertz)
			mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
			mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
				linear_to_mel_weight_matrix.shape[-1:]))
			log_offset = 1e-6
			log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
			mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., 1:2]
			
			features_t = mfccs
			#features_t = tf.concat([mfccs, tf.expand_dims(spectral_centroid, axis=2)], 2)
			#features_t1 = tf.expand_dims(spectral_centroid, axis=2)
			#features_t = tf.concat([features_t1, tf.multiply(0.0, features_t1)], 2)
			#features_t = mfccs
		with tf.name_scope('mean_var') as scope:
			features_mean, features_variance = tf.nn.moments(features_t, 1)
			z = tf.zeros([tf.shape(features_mean)[0], 1], tf.float32)
			#features = tf.concat([features_mean, features_variance], 1)
			features = tf.concat([features_mean, z, z], 1)
	return (n_mfccs, features)
	
def normalize_features_op(features):
	with tf.name_scope('norm_features') as scope:
		#batch_mean = tf.reduce_mean(features, axis=0, keep_dims=True)
		batch_mean, batch_variance = tf.nn.moments(features, axes=[0], keep_dims=True)
		normalized_features = tf.nn.batch_normalization(features, batch_mean, batch_variance, None, None, 1e-3)
		#normalized_features = features
	return normalized_features
	
def setup_features_graph():
	file, decoded_audio = read_audio_op()
	fr_length, fr_step, frames_flat = frame_op(decoded_audio)
	frames_normalized = normalize_audio_op(frames_flat)
	n_mfccs, features = features_op(frames_normalized)
	normalized_features = normalize_features_op(features)
	settings = {
		'fr_length': fr_length,
		'fr_step': fr_step,
		'n_mfccs': n_mfccs
	}
	return (settings, file, frames_normalized, normalized_features)

def setup_cluster_graph():
	with tf.name_scope('cluster') as scope:
		inp = tf.placeholder(tf.float32, name='feature_input')
		n_clusters = tf.placeholder(tf.int32, name='n_clusters')
		kmeans = tf.contrib.factorization.KMeans(inp, n_clusters, "kmeans_plus_plus")
		all_scores, cluster_idx, clustering_scores, initialized, init_op, training_op = kmeans.training_graph()
		#all_scores, cluster_idx, scores, training_op, init_op, initialized = tf.contrib.factorization.gmm(
		#	inp=inp, initial_clusters='random', num_clusters=20, random_seed=123)
		
		settings = {
			'inputs': inp,
			'n_clusters': n_clusters
		}
	return (settings, n_clusters, cluster_idx, initialized, init_op, training_op)

def write_cluster_file(filename, idx, frame_length, frame_step, sr):
	cluster_filename = os.path.splitext(filename)[0] + '.clusters.txt'
	with open(cluster_filename, 'w') as cluster_file:
		frame_length_seconds = frame_length / sr
		frame_step_seconds = frame_step / sr
		label_frame_start = 0
		label_cluster_id = -1
		for i in range(len(idx)):
			current_frame_start = i * frame_step_seconds
			current_frame_end = current_frame_start + frame_length_seconds
			current_cluster_id = idx[i]
			if label_cluster_id == -1:
				label_cluster_id = current_cluster_id
			else:
				label_frame_end = (i - 1) * frame_step_seconds + frame_length_seconds
				if current_cluster_id != label_cluster_id:
					cluster_file.write("%f\t%f\t%d\n" % (label_frame_start,
														 label_frame_end,
														 label_cluster_id))
					label_frame_start = label_frame_end
					label_cluster_id = current_cluster_id
		label_frame_end = (len(idx) - 1) * frame_step_seconds + frame_length_seconds
		cluster_file.write("%f\t%f\t%d" % (label_frame_start,
										   label_frame_end,
										   label_cluster_id))

def features_dict(s):
	return {
		s['fr_length']: 11025,
		s['fr_step']: 11025,
		s['n_mfccs']: 20
	}
	
def clusters_dict(s, inp):
	return {
		s['inputs']: inp,
		s['n_clusters']: 20
	}
										   
def main(_):
	with tf.Graph().as_default():
		features_settings, file, frames_normalized, normalized_features = setup_features_graph()
		clusters_settings, n_clusters, cluster_idx, initialized, kmeans_init, kmeans_training_op = setup_cluster_graph()
		init = tf.global_variables_initializer()
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with tf.Session(config = config) as sess:
			writer = tf.summary.FileWriter("d:\\users\\bruno\\aim\\tb")
			
			n_frames = []
			inputs_concat = np.array([])
			frames_concat = np.array([])
			print('Extracting features...')
			while True:
				try:
					frames, filename, inputs = sess.run([frames_normalized, file, normalized_features], features_dict(features_settings))
					inputs_concat = np.concatenate((
						inputs_concat, inputs)) if inputs_concat.size else inputs
					frames_concat = np.concatenate((frames_concat, frames)) if frames_concat.size else frames
					n_frames.append([filename.decode(), inputs.shape[0]])
					print('Processed input file %s' % filename.decode())
				except tf.errors.OutOfRangeError:
					break
			print('Extracting features...Done (frames, dimensions): %s' % str(inputs_concat.shape))
			
			print('Feature index, mean, standard deviation:')
			i = np.arange(inputs_concat.shape[1])
			m = np.mean(inputs_concat, axis=0)
			v = np.std(inputs_concat, axis=0)
			feature_info = np.transpose(np.vstack((i, m, v)))
			print(feature_info)
				
			sess.run(init)
			feed_dict = clusters_dict(clusters_settings, inputs_concat)
			sess.run(kmeans_init, feed_dict)
				
			print('Clustering...')
			for i in range(200):
				if (i % 10) == 0:
					print('%d...' % i)
				sess.run(kmeans_training_op, feed_dict)
			print('Clustering...Done')
			idx = sess.run(cluster_idx, feed_dict)
			#[0]

			number_of_frames = idx[0].shape[0]
			#print('Grouping %d frames by cluster...' % number_of_frames)
			#for cluster in range(20):
			#	print('Cluster %d...' % cluster)
			#	indices = np.where(idx[0] == cluster)
			#	frames = np.take(frames_concat, indices, axis=0)[0]
			#	if frames.size:
			#		print(frames.shape)
			#		random.shuffle(frames)
			#		v = tf.Variable(frames, dtype=tf.float32, name='vc_' + str(cluster))
			#		tf.summary.audio('cluster_' + str(cluster), v, 22050, max_outputs=10)
			#	else:
			#		print('(0, 0)')
			
			
		
			print('Grouping frames by cluster...Done')
			
			sess.run(tf.global_variables_initializer())
			#merged = tf.summary.merge_all()
			#summary = sess.run(merged)
			#writer.add_summary(summary)
			#writer.add_graph(sess.graph)
			#writer.flush()

			#with open(metadata, 'w') as metadata_file:
			#	for row in idx:
			#		metadata_file.write('%s\n' % row)
			with open(metadata, 'w') as metadata_file:
				metadata_file.write('File\tCluster\n')
				j = 0
				for row in n_frames:
					for i in range(row[1]):
						metadata_file.write('%s\t%d\n' % (row[0], idx[0][j]))
						j += 1
			
			features = tf.Variable(inputs_concat, name='features')
			
			saver = tf.train.Saver([features])
			sess.run(features.initializer)
			saver.save(sess, os.path.join(LOG_DIR, 'features.ckpt'))
			
			config = projector.ProjectorConfig()
			embedding = config.embeddings.add()
			embedding.tensor_name = features.name
			embedding.metadata_path = metadata
			projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

		# Iterate over input files and write cluster labels
		start_index = 0
		report = np.array([])
		for f in n_frames:
			end_index = start_index + f[1]
			file_clusters = idx[0][start_index:end_index]
			write_cluster_file(f[0], file_clusters, 11025, 11025, 22050)
			start_index += f[1]
		exit(0)
		with open('clusters' + str(int(time.time())) + '.csv', 'w') as file:
			for i in range(report.shape[1]):
				file.write(';%s' % i)
			file.write('\n')
			for i in range(report.shape[0]):
				file.write(n_frames[i][0])
				for j in range(report.shape[1]):
					file.write(';%s' % report[i][j])
				file.write('\n')
			file.write('sum')
			for i in range(report.shape[1]):
				file.write(';%s' % np.sum(report, axis=0)[i])
			file.write('\n')

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])