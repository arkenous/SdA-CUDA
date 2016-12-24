#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include "Data.h"
#include "Data_me.h"
#include "Data_other.h"
#include "StackedDenoisingAutoencoder.h"
#include "Normalize.h"
#include "AddNoise.h"

using std::vector;
using std::string;
using std::random_device;
using std::mt19937;
using std::cout;
using std::endl;
using std::shuffle;

int main() {
  double dropout_rate = 0.5;
  unsigned long num_sda_layer = 1;
  float sda_compression_rate = 0.3;
  cout << dropout_rate << " " << num_sda_layer << " " << sda_compression_rate << endl;

  random_device rnd;
  mt19937 mt;
  mt.seed(rnd());

  vector<vector<double>> noised = add_random_noise(me, 1.0);

  for (unsigned long i = 0, size = me.size(); i < size; ++i)
    normalize(&me[i]);
  for (unsigned long i = 0, size = noised.size(); i < size; ++i)
    normalize(&noised[i]);
  for (unsigned long i = 0, size = other.size(); i < size; ++i)
    normalize(&other[i]);

  // test_successのシャッフル
  shuffle(me.begin(), me.end(), mt);
  shuffle(noised.begin(), noised.end(), mt);
  shuffle(other.begin(), other.end(), mt);

  vector<vector<double>> train;
  train.push_back(me[0]);
  train.push_back(me[1]);
  train.push_back(me[2]);

  StackedDenoisingAutoencoder stackedDenoisingAutoencoder;
  stackedDenoisingAutoencoder.learn(train, num_sda_layer,
                                    sda_compression_rate, dropout_rate);

  for (unsigned long i = 0, size = me.size(); i < size; ++i) {
    vector<double> result = stackedDenoisingAutoencoder.out(me[i]);
    std::ofstream output_stream;
    output_stream.open("me_"+std::to_string(i)+".dat", std::ios::out);
    for (unsigned long j = 0, data_size = result.size(); j < data_size; ++j) {
      output_stream << result[j] << endl;
    }
  }

  return 0;
}

