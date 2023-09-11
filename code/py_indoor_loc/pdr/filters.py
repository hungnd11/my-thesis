import numpy as np
from scipy.ndimage import median_filter, uniform_filter


def filter_headings(data: np.ndarray, 
                    median_window_size: int = 9, 
                    mean_window_size: int = 3) -> np.ndarray:
  
  data_f = np.zeros_like(data, dtype=np.float32)
  data_f[:] = median_filter(data, size=median_window_size)
  data_f[:] = uniform_filter(data_f, size=mean_window_size)
  
  return data_f
