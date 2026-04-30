"""
Sampling and Anti-Aliasing Demonstration

This script reproduces a music sampling experiment similar to the MIT 6.003
Signals and Systems sampling demo.

It does the following:

1. Load an input violin WAV file.
2. Plot the original waveform.
3. Plot the original frequency spectrum.
4. Simulate sampling at:
   - 11 kHz
   - 5.5 kHz
   - 2.8 kHz
5. For each sampling rate, generate:
   - without anti-aliasing
   - with anti-aliasing
6. Save six playable WAV files.
7. Plot spectra of the six processed results.

Author: GPT5.5
"""

# sys is used to read command-line arguments.
import sys

# Path is used for convenient file and folder path handling.
from pathlib import Path

# numpy is used for numerical array operations.
import numpy as np

# soundfile is used to read and write WAV files.
import soundfile as sf

# scipy.signal provides filtering, resampling, and frequency analysis tools.
from scipy import signal

# matplotlib is used for plotting waveforms and spectra.
import matplotlib.pyplot as plt


# ============================================================
# 1. Global experiment configuration
# ============================================================

# The final WAV files are saved at this playback sample rate.
# Most audio players and sound cards handle 44100 Hz very well.
PLAYBACK_SR = 44100

# These are the three sampling frequencies shown in the lecture slide.
TARGET_SAMPLE_RATES = [11000, 5500, 2800]

# For anti-aliasing, we choose a cutoff slightly below the Nyquist frequency.
# Nyquist frequency is fs / 2.
# If cutoff is exactly fs / 2, a real filter cannot transition sharply enough.
# So we use 0.45 * fs, which is safely below fs / 2.
ANTI_ALIAS_CUTOFF_RATIO = 0.45

# Number of taps in the FIR anti-aliasing filter.
# Larger number means sharper transition band but slower computation.
FIR_NUM_TAPS = 801

# The output folder for generated WAV files and figures.
OUTPUT_DIR = Path("sampling_demo_outputs")


# ============================================================
# 2. Helper functions
# ============================================================

def ensure_output_dir():
    """
    Create the output folder if it does not already exist.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_audio_mono(input_path):
    """
    Load a WAV file and convert it to mono.

    Parameters
    ----------
    input_path : str or Path
        Path to the input WAV file.

    Returns
    -------
    x : numpy.ndarray
        Mono audio signal as a floating-point array.
    sr : int
        Original sample rate of the WAV file.
    """

    # Read audio data and sample rate.
    x, sr = sf.read(input_path)

    # If the audio has more than one channel, average all channels to mono.
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    # Convert audio to double precision floating point.
    x = x.astype(np.float64)

    # Normalize the audio to avoid clipping later.
    x = normalize_audio(x)

    return x, sr


def normalize_audio(x, peak=0.95):
    """
    Normalize audio so that its maximum absolute amplitude equals `peak`.

    Parameters
    ----------
    x : numpy.ndarray
        Input audio signal.
    peak : float
        Desired maximum absolute amplitude.

    Returns
    -------
    y : numpy.ndarray
        Normalized audio signal.
    """

    # Find the maximum absolute amplitude.
    max_abs = np.max(np.abs(x))

    # If the signal is almost silent, return it unchanged.
    if max_abs < 1e-12:
        return x

    # Scale the signal so that its maximum absolute value is `peak`.
    return peak * x / max_abs


def resample_audio(x, sr_in, sr_out):
    """
    Resample audio from sr_in to sr_out using scipy.signal.resample_poly.

    This function performs high-quality sample-rate conversion.
    Internally, resample_poly uses polyphase filtering.

    Parameters
    ----------
    x : numpy.ndarray
        Input audio signal.
    sr_in : int
        Input sample rate.
    sr_out : int
        Output sample rate.

    Returns
    -------
    y : numpy.ndarray
        Resampled audio signal.
    """

    # If the sample rates are the same, no resampling is needed.
    if sr_in == sr_out:
        return x

    # Compute greatest common divisor to reduce the up/down ratio.
    gcd = np.gcd(sr_in, sr_out)

    # Upsampling factor.
    up = sr_out // gcd

    # Downsampling factor.
    down = sr_in // gcd

    # Perform high-quality polyphase resampling.
    y = signal.resample_poly(x, up, down)

    return y


def plot_waveform(x, sr, title, save_path=None, max_seconds=0.05):
    """
    Plot the waveform of an audio signal.

    For readability, only the first `max_seconds` seconds are plotted.

    Parameters
    ----------
    x : numpy.ndarray
        Audio signal.
    sr : int
        Sample rate.
    title : str
        Figure title.
    save_path : str or Path or None
        If provided, save the figure to this path.
    max_seconds : float
        Number of seconds to display.
    """

    # Decide how many samples to show.
    n_show = min(len(x), int(max_seconds * sr))

    # Create a time axis in seconds.
    t = np.arange(n_show) / sr

    # Create a new figure.
    plt.figure(figsize=(10, 4))

    # Plot amplitude versus time.
    plt.plot(t, x[:n_show])

    # Label the horizontal axis.
    plt.xlabel("Time (s)")

    # Label the vertical axis.
    plt.ylabel("Amplitude")

    # Add title.
    plt.title(title)

    # Add grid for readability.
    plt.grid(True)

    # Make the layout compact.
    plt.tight_layout()

    # Save if requested.
    if save_path is not None:
        plt.savefig(save_path, dpi=160)

    # Display the figure.
    plt.show()


def compute_single_sided_spectrum(x, sr, max_freq=None):
    """
    Compute the single-sided amplitude spectrum of an audio signal.

    This is useful for visualizing the finite frequency range of the audio.
    In theory, audio sampled at sr has spectrum from 0 to sr / 2.

    Parameters
    ----------
    x : numpy.ndarray
        Audio signal.
    sr : int
        Sample rate.
    max_freq : float or None
        Maximum frequency to return. If None, use sr / 2.

    Returns
    -------
    freqs : numpy.ndarray
        Frequency axis in Hz.
    magnitude_db : numpy.ndarray
        Magnitude spectrum in dB.
    """

    # Use a limited segment for spectrum analysis to keep FFT efficient.
    # Here we analyze at most the first 10 seconds.
    max_samples = min(len(x), int(10 * sr))

    # Take the first segment of audio.
    x_seg = x[:max_samples]

    # Apply a Hann window to reduce spectral leakage.
    window = np.hanning(len(x_seg))

    # Windowed signal.
    x_win = x_seg * window

    # Compute real FFT because audio signal is real-valued.
    X = np.fft.rfft(x_win)

    # Compute corresponding frequency bins.
    freqs = np.fft.rfftfreq(len(x_win), d=1.0 / sr)

    # Convert magnitude to dB.
    # The small epsilon avoids log of zero.
    magnitude_db = 20 * np.log10(np.abs(X) + 1e-12)

    # If max_freq is not provided, use Nyquist frequency.
    if max_freq is None:
        max_freq = sr / 2

    # Keep only frequencies up to max_freq.
    mask = freqs <= max_freq

    return freqs[mask], magnitude_db[mask]


def plot_spectrum(x, sr, title, save_path=None, max_freq=12000):
    """
    Plot the single-sided spectrum of an audio signal.

    Parameters
    ----------
    x : numpy.ndarray
        Audio signal.
    sr : int
        Sample rate.
    title : str
        Figure title.
    save_path : str or Path or None
        If provided, save the figure.
    max_freq : float
        Maximum frequency shown in the plot.
    """

    # Do not plot beyond the signal's Nyquist frequency.
    effective_max_freq = min(max_freq, sr / 2)

    # Compute frequency axis and magnitude spectrum.
    freqs, magnitude_db = compute_single_sided_spectrum(
        x,
        sr,
        max_freq=effective_max_freq
    )

    # Create a new figure.
    plt.figure(figsize=(10, 4))

    # Plot magnitude in dB versus frequency.
    plt.plot(freqs, magnitude_db)

    # Label the horizontal axis.
    plt.xlabel("Frequency (Hz)")

    # Label the vertical axis.
    plt.ylabel("Magnitude (dB)")

    # Add title.
    plt.title(title)

    # Add grid.
    plt.grid(True)

    # Make layout compact.
    plt.tight_layout()

    # Save if requested.
    if save_path is not None:
        plt.savefig(save_path, dpi=160)

    # Display figure.
    plt.show()


def design_anti_aliasing_filter(sr_orig, sr_target):
    """
    Design an FIR low-pass anti-aliasing filter.

    In a real sampling system, the anti-aliasing filter is placed before
    the sampler. Its job is to remove frequency components above the
    target Nyquist frequency, fs_target / 2.

    Parameters
    ----------
    sr_orig : int
        Original high sample rate.
    sr_target : int
        Target lower sample rate.

    Returns
    -------
    taps : numpy.ndarray
        FIR filter coefficients.
    cutoff_hz : float
        Filter cutoff frequency in Hz.
    """

    # Choose cutoff below target Nyquist frequency.
    cutoff_hz = ANTI_ALIAS_CUTOFF_RATIO * sr_target

    # Original Nyquist frequency.
    nyquist_orig = sr_orig / 2

    # Normalize cutoff frequency to [0, 1], where 1 corresponds to sr_orig / 2.
    normalized_cutoff = cutoff_hz / nyquist_orig

    # Design a low-pass FIR filter using a Hamming window.
    taps = signal.firwin(
        FIR_NUM_TAPS,
        cutoff=normalized_cutoff,
        window="hamming"
    )

    return taps, cutoff_hz


def apply_anti_aliasing_filter(x, sr_orig, sr_target):
    """
    Apply anti-aliasing low-pass filter before sampling.

    Parameters
    ----------
    x : numpy.ndarray
        Original high-sample-rate signal.
    sr_orig : int
        Original sample rate.
    sr_target : int
        Target sampling rate.

    Returns
    -------
    y : numpy.ndarray
        Low-pass-filtered signal.
    cutoff_hz : float
        Cutoff frequency used.
    """

    # Design the FIR low-pass filter.
    taps, cutoff_hz = design_anti_aliasing_filter(sr_orig, sr_target)

    # Apply zero-phase filtering.
    # filtfilt filters forward and backward, so there is no phase delay.
    y = signal.filtfilt(taps, [1.0], x)

    return y, cutoff_hz


def idealized_sampler_by_interpolation(x, sr_orig, sr_target):
    """
    Simulate sampling at sr_target from a high-rate signal x.

    This function samples the original signal at uniformly spaced time instants
    corresponding to the target sampling rate.

    In continuous-time theory, sampling means multiplying by an impulse train.
    In a digital simulation, we cannot create true impulses. Instead, we assume
    the original high-rate WAV approximates a continuous-time signal, and we
    read its value at the desired sampling instants.

    Parameters
    ----------
    x : numpy.ndarray
        Original high-sample-rate audio.
    sr_orig : int
        Original sample rate.
    sr_target : int
        Target sampling rate.

    Returns
    -------
    y_low_sr : numpy.ndarray
        Samples at target sample rate.
    """

    # Compute the duration of the original audio in seconds.
    duration = len(x) / sr_orig

    # Compute the number of target samples.
    n_target = int(np.floor(duration * sr_target))

    # Time instants at which the low-rate sampler takes samples.
    t_target = np.arange(n_target) / sr_target

    # Original sample time axis.
    t_orig = np.arange(len(x)) / sr_orig

    # Interpolate the high-rate signal at target sampling instants.
    # This approximates sampling a continuous-time signal.
    y_low_sr = np.interp(t_target, t_orig, x)

    return y_low_sr


def reconstruct_for_playback(y_low_sr, sr_target, playback_sr=PLAYBACK_SR):
    """
    Convert the low-sample-rate signal back to 44100 Hz for playback.

    Important:
    The signal has already been sampled at sr_target.
    Resampling it back to 44100 Hz does not undo aliasing.
    It only makes the file playable by normal audio software.

    Parameters
    ----------
    y_low_sr : numpy.ndarray
        Low-sample-rate sampled audio.
    sr_target : int
        Sampling rate of y_low_sr.
    playback_sr : int
        Desired output WAV sample rate.

    Returns
    -------
    y_playback : numpy.ndarray
        Audio resampled to playback_sr.
    """

    # Use high-quality resampling for playback convenience.
    y_playback = resample_audio(y_low_sr, sr_target, playback_sr)

    # Normalize to avoid clipping.
    y_playback = normalize_audio(y_playback)

    return y_playback


def process_without_antialiasing(x, sr_orig, sr_target):
    """
    Simulate the system without anti-aliasing.

    System diagram:

        input signal x(t)
              |
              v
        sampler at fs = sr_target
              |
              v
        discrete-time low-rate signal
              |
              v
        resample to 44.1 kHz for playback

    In this case, high-frequency components above sr_target / 2 are not removed.
    Therefore, they alias into lower frequencies.

    Parameters
    ----------
    x : numpy.ndarray
        Original high-sample-rate audio.
    sr_orig : int
        Original sample rate.
    sr_target : int
        Target sample rate.

    Returns
    -------
    y_low_sr : numpy.ndarray
        Low-rate sampled signal.
    y_playback : numpy.ndarray
        44.1 kHz playable version.
    """

    # Directly sample the original signal at the lower sampling rate.
    y_low_sr = idealized_sampler_by_interpolation(x, sr_orig, sr_target)

    # Convert the low-rate result back to 44.1 kHz for playback.
    y_playback = reconstruct_for_playback(y_low_sr, sr_target)

    return y_low_sr, y_playback


def process_with_antialiasing(x, sr_orig, sr_target):
    """
    Simulate the system with anti-aliasing.

    System diagram:

        input signal x(t)
              |
              v
        analog low-pass anti-aliasing filter
              |
              v
        sampler at fs = sr_target
              |
              v
        discrete-time low-rate signal
              |
              v
        resample to 44.1 kHz for playback

    In a real ADC system, the anti-aliasing filter is analog.
    In this simulation, the original WAV is already digital, so we approximate
    that analog filter using a digital FIR low-pass filter before downsampling.

    Parameters
    ----------
    x : numpy.ndarray
        Original high-sample-rate audio.
    sr_orig : int
        Original sample rate.
    sr_target : int
        Target sample rate.

    Returns
    -------
    y_filtered : numpy.ndarray
        High-rate signal after anti-aliasing filter.
    y_low_sr : numpy.ndarray
        Low-rate sampled signal.
    y_playback : numpy.ndarray
        44.1 kHz playable version.
    cutoff_hz : float
        Anti-aliasing cutoff frequency.
    """

    # First, apply low-pass filtering before sampling.
    y_filtered, cutoff_hz = apply_anti_aliasing_filter(
        x,
        sr_orig,
        sr_target
    )

    # Then sample the filtered signal at the lower sampling rate.
    y_low_sr = idealized_sampler_by_interpolation(
        y_filtered,
        sr_orig,
        sr_target
    )

    # Convert the low-rate result back to 44.1 kHz for playback.
    y_playback = reconstruct_for_playback(y_low_sr, sr_target)

    return y_filtered, y_low_sr, y_playback, cutoff_hz


def save_wav(path, x, sr):
    """
    Save an audio signal as a WAV file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    x : numpy.ndarray
        Audio signal.
    sr : int
        Sample rate.
    """

    # Normalize before saving to avoid clipping.
    x = normalize_audio(x)

    # Write WAV file.
    sf.write(path, x, sr)


def print_explanation_for_case(sr_target):
    """
    Print theoretical notes for a given sampling frequency.

    Parameters
    ----------
    sr_target : int
        Target sampling frequency.
    """

    # Nyquist frequency is half of sampling frequency.
    nyquist = sr_target / 2

    print()
    print("=" * 72)
    print(f"Sampling frequency: fs = {sr_target} Hz")
    print(f"Nyquist frequency: fs / 2 = {nyquist:.1f} Hz")
    print("Without anti-aliasing:")
    print("  Frequency components above Nyquist will fold into lower frequencies.")
    print("With anti-aliasing:")
    print("  A low-pass filter is applied before sampling.")
    print("  This removes high-frequency content that would otherwise alias.")
    print("=" * 72)
    print()


# ============================================================
# 3. Main experiment
# ============================================================

def main(input_path):
    """
    Run the full sampling and anti-aliasing experiment.

    Parameters
    ----------
    input_path : str or Path
        Path to the input violin WAV file.
    """

    # Create output folder.
    ensure_output_dir()

    # Load input audio and convert to mono.
    x, sr_orig = load_audio_mono(input_path)

    # Print basic information about the input audio.
    print(f"Loaded input file: {input_path}")
    print(f"Original sample rate: {sr_orig} Hz")
    print(f"Number of samples: {len(x)}")
    print(f"Duration: {len(x) / sr_orig:.2f} seconds")

    # If the input sample rate is not 44100 Hz, convert it to 44100 Hz first.
    # This makes the rest of the experiment consistent.
    if sr_orig != PLAYBACK_SR:
        print(f"Resampling input from {sr_orig} Hz to {PLAYBACK_SR} Hz...")
        x = resample_audio(x, sr_orig, PLAYBACK_SR)
        sr_orig = PLAYBACK_SR
        x = normalize_audio(x)

    # Plot original waveform.
    plot_waveform(
        x,
        sr_orig,
        title="Original Violin Waveform",
        save_path=OUTPUT_DIR / "original_waveform.png",
        max_seconds=0.05
    )

    # Plot original spectrum.
    plot_spectrum(
        x,
        sr_orig,
        title="Original Violin Spectrum",
        save_path=OUTPUT_DIR / "original_spectrum.png",
        max_freq=12000
    )

    # Save normalized original for reference.
    save_wav(
        OUTPUT_DIR / "original_normalized.wav",
        x,
        sr_orig
    )

    # Process each target sample rate.
    for sr_target in TARGET_SAMPLE_RATES:

        # Print theoretical explanation.
        print_explanation_for_case(sr_target)

        # ------------------------------------------------------------
        # Case 1: without anti-aliasing
        # ------------------------------------------------------------

        # Simulate direct low-rate sampling without prefiltering.
        y_low_noaa, y_play_noaa = process_without_antialiasing(
            x,
            sr_orig,
            sr_target
        )

        # Output file name for the playable 44.1 kHz WAV.
        out_noaa = OUTPUT_DIR / f"violin_fs{sr_target}_without_antialiasing.wav"

        # Save the no-anti-aliasing result.
        save_wav(out_noaa, y_play_noaa, PLAYBACK_SR)

        # Also save the native low-rate sampled signal.
        # This file has actual sample rate sr_target.
        out_noaa_native = OUTPUT_DIR / f"violin_fs{sr_target}_without_antialiasing_native.wav"
        save_wav(out_noaa_native, y_low_noaa, sr_target)

        # Plot spectrum of the playable result.
        plot_spectrum(
            y_play_noaa,
            PLAYBACK_SR,
            title=f"fs = {sr_target} Hz, Without Anti-Aliasing",
            save_path=OUTPUT_DIR / f"spectrum_fs{sr_target}_without_antialiasing.png",
            max_freq=12000
        )

        # ------------------------------------------------------------
        # Case 2: with anti-aliasing
        # ------------------------------------------------------------

        # Apply anti-aliasing filter first, then sample.
        y_filtered, y_low_aa, y_play_aa, cutoff_hz = process_with_antialiasing(
            x,
            sr_orig,
            sr_target
        )

        # Print filter information.
        print(f"Anti-aliasing cutoff for fs = {sr_target} Hz: {cutoff_hz:.1f} Hz")

        # Output file name for the playable 44.1 kHz WAV.
        out_aa = OUTPUT_DIR / f"violin_fs{sr_target}_with_antialiasing.wav"

        # Save the anti-aliased result.
        save_wav(out_aa, y_play_aa, PLAYBACK_SR)

        # Also save the native low-rate sampled signal.
        out_aa_native = OUTPUT_DIR / f"violin_fs{sr_target}_with_antialiasing_native.wav"
        save_wav(out_aa_native, y_low_aa, sr_target)

        # Plot spectrum of the playable anti-aliased result.
        plot_spectrum(
            y_play_aa,
            PLAYBACK_SR,
            title=f"fs = {sr_target} Hz, With Anti-Aliasing",
            save_path=OUTPUT_DIR / f"spectrum_fs{sr_target}_with_antialiasing.png",
            max_freq=12000
        )

    # Final message.
    print()
    print("Done.")
    print(f"All WAV files and figures are saved in: {OUTPUT_DIR.resolve()}")


# ============================================================
# 4. Command-line entry point
# ============================================================

if __name__ == "__main__":

    # Check whether the user provided an input WAV file path.
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python sampling_anti_aliasing_demo.py input_violin.wav")
        sys.exit(1)

    # Read the input path from the command line.
    input_wav_path = sys.argv[1]

    # Run the experiment.
    main(input_wav_path)
