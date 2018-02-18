<h1 id="aim_ops">aim_ops</h1>

AIM-specific TensorFlow operations

<h2 id="aim_ops.read_audio">read_audio</h2>

```python
read_audio(file_pattern, name=None)
```
Read next audio file matching filename patterns and return decoded audio
    using an iterator.

  Args:
    file_pattern: A string or scalar string `Tensor`, representing the
    filename pattern that will be matched.

    name: `string`, name of the operation.

  Returns:
    An initializer operation for the iterator
    A `Tensor` containing the name of the decoded file.
    A `Tensor` containing the decoded audio.

<h2 id="aim_ops.read_ground_truth_labels">read_ground_truth_labels</h2>

```python
read_ground_truth_labels(audio_filename, fr_length, sample_rate, name=None)
```
Read the corresponding ground truth labels file for the given audio file.
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

<h2 id="aim_ops.remove_dc">remove_dc</h2>

```python
remove_dc(frames, name=None)
```
Remove dc offset from a batch of audio signals.

  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    name: `string`, name of the operation.

  Returns:
    A `Tensor` with the same shape as the input.

<h2 id="aim_ops.scale">scale</h2>

```python
scale(frames, attenuation, name=None)
```
Scale a batch of audio signals using root mean square normalization.

  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    attenuation: `float32`, the attenuation in decibel.
    name: `string`, name of the operation.

  Returns:
    A `Tensor` with the same shape as the input.

<h2 id="aim_ops.normalize_audio">normalize_audio</h2>

```python
normalize_audio(frames, attenuation, name=None)
```
Normalize a batch of audio signals by removing dc offset and applying
    rms normalization.

  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    attenuation: `float32`, the attenuation in decibel.
    name: `string`, name of the operation.

  Returns:
    A `Tensor` with the same shape as the input.

<h2 id="aim_ops.mag_spectrogram">mag_spectrogram</h2>

```python
mag_spectrogram(frames, fft_length=1024, fft_step=512, name=None)
```
Extract magnitude spectrograms from a batch of audio signals.

  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    fft_length: An integer scalar `Tensor`. The window length in samples.
    fft_step: An integer scalar `Tensor`. The number of samples to step.
    name: `string`, name of the operation.

  Returns:
    A `Tensor` with shape `[frames, spectrogram_bins]`.

<h2 id="aim_ops.log_mel_spectrogram">log_mel_spectrogram</h2>

```python
log_mel_spectrogram(mag_spectrogram, sample_rate, lower_edge_hertz, upper_edge_hertz, n_mel_bins=128, name=None)
```
Calculate log mel spectrograms from a batch of magnitude spectrograms.

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

<h2 id="aim_ops.mfcc">mfcc</h2>

```python
mfcc(log_mel_spectrogram, n_mfcc=20, name=None)
```
Calculate mel frequency cepstral coefficients from a batch of log mel
    spectrograms.

  Args:
    log_mel_spectrogram: A `Tensor` of shape `[frames, mel_bins]`.
    n_mfcc: The number of coefficients to extract.

  Returns:
    A `Tensor` with shape `[frames, n_mfcc]` containing the mfccs.
    A `Tensor` with shape `[n_mfcc]` containing the feature ids.

<h2 id="aim_ops.spectral_centroid_bandwidth">spectral_centroid_bandwidth</h2>

```python
spectral_centroid_bandwidth(mag_spectrogram, sample_rate, p, name=None)
```
Calculate the logarithmized spectral centroid from a batch of magnitude
    spectrograms.

  Args:
    mag_spectrogram: A `Tensor` of shape `[frames, spectrogram_bins]`.
    sample_rate: Python float. Samples per second of the input signal.
    p: Power to raise deviation from spectral centroid.
    name: `string`, name of the operation.

  Returns:
    A `Tensor` with shape `[frames, 1]` containing the spectral centroid.
    A `Tensor` with shape `[1]` containing the feature id.

<h2 id="aim_ops.zero_crossing_rate">zero_crossing_rate</h2>

```python
zero_crossing_rate(frames, sfr_length=1024, sfr_step=512, name=None)
```
Calculate the zero crossing rate for a batch of audio signals.

  Args:
    frames: A `Tensor` of shape `[frames, samples]`.
    sfr_length: An integer scalar `Tensor`. The subframe length in samples.
    sfr_step: An integer scalar `Tensor`. The number of samples to step.
    name: `string`, name of the operation.

  Returns:
    A `Tensor` with shape `[frames, 1]` containing the zero crossing rate.
    A `Tensor` with shape `[1]` containing the feature id.

<h2 id="aim_ops.extract_features">extract_features</h2>

```python
extract_features(frames, sample_rate, lower_edge_hertz, upper_edge_hertz, sfr_length=1024, sfr_step=512, n_mel_bins=128, n_mfcc=20, p=2.0, name=None)
```
Extract a batch of features from a batch of audio signals.

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

<h2 id="aim_ops.reduce_features">reduce_features</h2>

```python
reduce_features(features, feature_ids, mean=True, variance=True, name=None)
```
Reduce a batch of features along the time axis for each frame, calculating
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

<h2 id="aim_ops.normalize_features">normalize_features</h2>

```python
normalize_features(features, name=None)
```
Normalize a batch of features to zero mean, unit variance.

  Args:
    features: A `Tensor` of shape `[frames, features]`.
    name: `string`, name of the operation.

  Returns:
    A `Tensor` of shape `[frames, features]`.

