"""
Useful annotations
"""
import time


def override(f):
  return f


def execution_time(func):

  def wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    ans = func(*args, **kwargs)
    finish_time = time.perf_counter()
    return ans, finish_time - start_time

  return wrapper
