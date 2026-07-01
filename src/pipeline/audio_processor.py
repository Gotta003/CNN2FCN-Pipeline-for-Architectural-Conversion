from __future__ import annotations
import numpy as np
import warnings
try:
    import cupy as cp
    from cupy.fft import rfft as xp_rfft
    HAS_GPU=True
except ImportError:
    import numpy as cp
    from numpy.fft import rfft as xp_rfft
    HAS_GPU=False
    
try:
    if HAS_GPU:
        import cupyx.scipy.fftpack as xp_dct
    else:
        import scipy.fftpack as xp_dct
except ImportError:
    import scipy.fftpack as xp_dct
    
warnings.filterwarnings("ignore", category=UserWarning)

#Constants
SAMPLE_RATE=16000
FRAME_DUR=0.032
FRAME_SIZE=int(SAMPLE_RATE*FRAME_DUR)
FRAME_STRIDE_DUR=0.024
FRAME_STRIDE=int(SAMPLE_RATE*FRAME_STRIDE_DUR)
NUM_BINS=FRAME_SIZE//2
FILTER_NUMBER=40
MIN_FREQ=0
MAX_FREQ=SAMPLE_RATE//2
COEFFICIENT=0.96875
NOISE_FLOOR=-40.0
TARGET_FRAMES=40
NUM_MFCC=13

class AudioProcessor:
    def __init__(self, use_gpu: bool=True):
        self.device_gpu=HAS_GPU and use_gpu
        self.xp=cp if self.device_gpu else np
        self.rfft=xp_rfft
        self.window=self._get_hamming_window(FRAME_SIZE)
        self.filterbank=self._create_mel_filterbank()
        
    def _get_hamming_window(self, size: int):
        return 0.54-0.46*self.xp.cos(2*self.xp.pi*self.xp.arange(size)/(size-1))
    
    def _hz_to_mel(self, hz):
        return 1127.0*self.xp.log10(1+hz/700.0)
    
    def _mel_to_hz(self, mel):
        return 700*(10**(mel/1127.0)-1)
    
    def _create_mel_filterbank(self):
        min_mel=self._hz_to_mel(MIN_FREQ)
        max_mel=self._hz_to_mel(MAX_FREQ)
        mel_points=self.xp.zeros(FILTER_NUMBER+2)
        mel_spacing=(max_mel-min_mel)/(FILTER_NUMBER+1)
        for i in range(FILTER_NUMBER+2):
            mel_points[i]=self._mel_to_hz(min_mel+i*mel_spacing)
        mel_points=self.xp.clip(mel_points, 0, MAX_FREQ)
        bin_indices=self.xp.array(mel_points*(NUM_BINS-1)/(SAMPLE_RATE/2.0), dtype=self.xp.int32)
        bin_indices=self.xp.clip(bin_indices, 0, NUM_BINS-1)
        filterbank=self.xp.zeros((FILTER_NUMBER, NUM_BINS), dtype=self.xp.float32)
        for i in range(FILTER_NUMBER):
            left, middle, right=int(bin_indices[i]), int(bin_indices[i+1]), int(bin_indices[i+2])
            if left==middle:
                middle=min(left+1, NUM_BINS-1)
            if middle==right:
                right=min(middle+1, NUM_BINS-1)
            for j in range(left, middle):
                filterbank[i, j]=(j-left)/(middle-left)                                                             
            for j in range(middle, right):
                filterbank[i, j]=1.0-(j-middle)/(right-middle)
        return filterbank
    
    def _pre_emphasize(self, audio_cpu):
        audio_scaled=audio_cpu/32768.0
        x=self.xp.asarray(audio_scaled, dtype=self.xp.float32)
        emphasized=self.xp.zeros_like(x)
        emphasized[0]=x[0]
        emphasized[1:]=x[1:]-COEFFICIENT*x[:-1]
        return emphasized

    def compute_features(self, audio_cpu, feature_type: str="mfe") -> np.ndarray:
        pre_emphasized=self._pre_emphasize(audio_cpu)
        num_frames_available=int((len(pre_emphasized)-FRAME_SIZE)/FRAME_STRIDE)+1
        num_frames=min(num_frames_available, TARGET_FRAMES)
        spectrogram=self.xp.zeros((num_frames, NUM_BINS), dtype=self.xp.float32)
        for frame in range(num_frames):
            start=frame*FRAME_STRIDE
            end=start+FRAME_SIZE
            segment=pre_emphasized[start:end]
            if len(segment)<FRAME_SIZE:
                segment=self.xp.pad(segment, (0, FRAME_SIZE-len(segment)))
            windowed=segment*self.window
            fft=self.rfft(windowed, n=FRAME_SIZE)
            spectrogram[frame]=self.xp.abs(fft)[:NUM_BINS]
        mel_spec=self.xp.dot(spectrogram, self.filterbank.T)
        log_mel=10*self.xp.log10(mel_spec+1e-20)
        log_mel=(log_mel-NOISE_FLOOR)/(-NOISE_FLOOR+12)
        quantized=self.xp.clip(self.xp.round(log_mel*256)/256.0, 0, 1)
        if feature_type.upper()=="mfcc":
            if self.device_gpu:
                dct_out=xp_dct.dct(quantized, type=2, axis=1, norm="ortho")
            else:
                import scipy.fftpack as mf_dct
                dct_out=mf_dct.dct(quantized, type=2, axis=1, norm="ortho")
            quantized=dct_out[:, :NUM_MFCC]
        if quantized.shape[0]<TARGET_FRAMES:
            quantized=self.xp.pad(quantized, ((0, TARGET_FRAMES-quantized.shape[0]), (0, 0)))
        else:
            quantized=quantized[:TARGET_FRAMES]
        return quantized.get() if self.device_gpu else quantized