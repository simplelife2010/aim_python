import tensorflow as tf
import aim_ops as aim

class TestAimOps(tf.test.TestCase):

  def test_read_audio(self):
    with self.test_session():
      temp_dir = tf.test.get_temp_dir()
      wav_filename = temp_dir + "\\test.wav"
      with open(wav_filename, "wb") as wav_file:
        wav_file.write(bytearray.fromhex("524946467C0000005741564566"))
        wav_file.write(bytearray.fromhex("6D742010000000010001002256"))
        wav_file.write(bytearray.fromhex("000044AC000002001000646174"))
        wav_file.write(bytearray.fromhex("610E00000093021D02440066FE"))
        wav_file.write(bytearray.fromhex("13FE81FEF9FE4C495354180000"))
        wav_file.write(bytearray.fromhex("00494E464F494E414D02000000"))
        wav_file.write(bytearray.fromhex("35004954524B02000000310069"))
        wav_file.write(bytearray.fromhex("64332022000000494433030000"))
        wav_file.write(bytearray.fromhex("000000185452434B0000000200"))
        wav_file.write(bytearray.fromhex("00003154495432000000020000"))
        wav_file.write(bytearray.fromhex("0035"))
      initializer, next_filename, decoded_audio = aim.read_audio(
        temp_dir + "\\*.wav")
      initializer.run()
      self.assertEqual(next_filename.eval(), wav_filename.encode())
      initializer.run()
      self.assertAllClose(decoded_audio.eval(), [
        0.02011108,
        0.01651001,
        0.0020752,
        -0.01251221,
        -0.01504517,
        -0.01168823,
        -0.00802612
        ])
  
  def test_read_ground_truth_labels(self):
    with self.test_session():
      temp_dir = tf.test.get_temp_dir()
      audio_filename = temp_dir + "\\test.wav"
      gt_filename = temp_dir + "\\test.gt"
      with open(gt_filename, "w") as gt_file:
        gt_file.write("0.000000\t1.500000\t1\r\n1.500000\t2.500000\t2")
      labels_by_frame = aim.read_ground_truth_labels(
        audio_filename, 11025, 22050)
      self.assertAllEqual(labels_by_frame.eval(), [1, 1, 1, 2, 2])

  def test_remove_dc(self):
    with self.test_session():
      signal_with_dc = [
        [1.0, 0.0, 0.5],
        [-2.0, 1.0, 1.6]
        ]
      signal_without_dc = aim.remove_dc(signal_with_dc)
      self.assertAllClose(signal_without_dc.eval(), [
        [0.5, -0.5, 0.0],
        [-2.2, 0.8, 1.4]
        ])
  
  def test_scale(self):
    with self.test_session():
      unscaled_signal = [
        [0.0, -0.5, 0.0, 0.5],
        [1.0, 0.0, -1.0, -2.0]
      ]
      scaled_signal = aim.scale(unscaled_signal, 12)
      self.assertAllClose(scaled_signal.eval(), [
        [0, -0.3552343859, 0.0, 0.3552343859],
        [0.2050946683, 0.0, -0.2050946683, -0.4101893366]
        ])
  
  def test_normalize_audio(self):
    with self.test_session():
      signal_with_dc = [
        [1.0, 0.2, 0.5, -3.1],
        [-2.0, 1.0, 1.6, 4.4]
        ]
      normalized_signal = aim.normalize_audio(signal_with_dc, 18)
      self.assertAllClose(normalized_signal.eval(), [
        [
      
if __name__ == "__main__":
  tf.test.main()