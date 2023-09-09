"""
kNN experiment
"""
import numpy as np

from py_indoor_loc.knn.preprocessing import min_filter
from sklearn.neighbors import KNeighborsRegressor
from typing import Iterable
from tqdm import tqdm


def report_localization_error(errors: np.ndarray) -> dict:
  assert len(errors) > 0

  result = {
      "count": errors.shape[0],
      "mean": errors.mean(),
      "std": errors.std(),
  }

  for p in [5, 10, 25, 50, 75, 90, 95]:
    result[f"p{p}"] = np.percentile(errors, p)

  return result


def run_knn_regression(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       knn_kwargs: dict) -> dict:
  knn = KNeighborsRegressor(**knn_kwargs)
  _ = knn.fit(X_train, y_train)

  y_pred = knn.predict(X_test)
  errors = np.linalg.norm(y_test - y_pred, axis=1)

  return report_localization_error(errors)


def run_knn_regression_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    options_min_rss: Iterable[float] = (-100, -95, -90, -85, -80, -75),
    options_metric: Iterable[str] = ("l1", "l2", "cosine"),
    options_n_neighbors: Iterable[int] = (1, 2, 3, 4, 5, 6, 7, 8)
) -> list[dict]:

  options = []
  for min_rss in options_min_rss:
    for metric in options_metric:
      for n_neighbors in options_n_neighbors:
        options.append((min_rss, metric, n_neighbors))

  results = []
  for (min_rss, metric, n_neighbors) in tqdm(options):
    X_train_t = min_filter(X_train, min_rss)
    X_test_t = min_filter(X_test, min_rss)
    knn_kwargs = {"n_neighbors": n_neighbors, "metric": metric}
    result = run_knn_regression(X_train_t, y_train, X_test_t, y_test,
                                knn_kwargs)
    result.update({
        "param_min_rss": min_rss,
        "param_metric": metric,
        "param_n_neighbors": n_neighbors,
    })
    results.append(result)

  return results
