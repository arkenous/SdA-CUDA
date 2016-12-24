
#ifndef SDA_CUDA_COSSIMILARITY_H
#define SDA_CUDA_COSSIMILARITY_H

#include <vector>
#include <cmath>

using std::vector;
class CosSimilarity {
public:
  double cos_similarity(vector<double> &a, vector<double> &b);

private:
  double similarity = 0.0;
  double ab = 0.0;
  double size_a = 0.0;
  double size_b = 0.0;
};

#endif //SDA_CUDA_COSSIMILARITY_H
