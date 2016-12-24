
#include "CosSimilarity.h"

using std::sqrt;
using std::pow;

double CosSimilarity::cos_similarity(vector<double> &a, vector<double> &b) {
  for (unsigned long i = 0, size = a.size(); i < size; ++i) {
    ab += a[i] * b[i];
    size_a += sqrt(pow(a[i], 2.0));
    size_b += sqrt(pow(b[i], 2.0));
  }

  return ab / (size_a * size_b);
}