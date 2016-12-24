
#ifndef SDA_CUDA_STACKEDDENOISINGAUTOENCODER_H
#define SDA_CUDA_STACKEDDENOISINGAUTOENCODER_H

#include <vector>
#include <string>
#include <thread>
#include <zconf.h>
#include "Neuron.cuh"

using std::string;
using std::vector;
using std::thread;

class StackedDenoisingAutoencoder {
public:
  StackedDenoisingAutoencoder();

  void learn(const vector<vector<double>> &input, const unsigned long result_num_dimen,
                    const float compression_rate, const double dropout_rate);
  vector<double> out(const vector<double> &input);

private:
  unsigned long num_thread = (unsigned long) sysconf(_SC_NPROCESSORS_ONLN);
  unsigned long num_middle_neurons;

  vector<vector<Neuron>> sda_neurons;
  vector<vector<double>> sda_out;

  vector<thread> threads;
  vector<double> in;

  void sdaFirstLayerOutThread(const int begin, const int end);
  void sdaOtherLayerOutThread(const int layer, const int begin, const int end);
};

#endif //SDA_CUDA_STACKEDDENOISINGAUTOENCODER_H
