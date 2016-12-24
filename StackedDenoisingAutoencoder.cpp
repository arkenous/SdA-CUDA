
#include <random>
#include <sstream>
#include <iostream>
#include "StackedDenoisingAutoencoder.h"
#include "DenoisingAutoencoder.h"
#include "AddNoise.h"

using std::string;
using std::vector;
using std::stringstream;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;
using std::thread;

StackedDenoisingAutoencoder::StackedDenoisingAutoencoder() {
  threads = vector<thread>(num_thread);
}


void StackedDenoisingAutoencoder::learn(const vector<vector<double>> &input,
                                          const unsigned long result_num_layer,
                                          const float compression_rate,
                                          const double dropout_rate) {
  sda_neurons.resize(result_num_layer);
  sda_out.resize(result_num_layer);
  unsigned long num_sda_layer = 0;

  vector<vector<double>> answer(input);
  vector<vector<double>> noisy_input(add_noise(input, 0.5));

  DenoisingAutoencoder denoisingAutoencoder(noisy_input[0].size(), compression_rate, dropout_rate);

  // Store learned dA middle neurons
  sda_neurons[num_sda_layer] = denoisingAutoencoder.learn(answer, noisy_input);
  sda_out[num_sda_layer].resize(sda_neurons[num_sda_layer].size());

  num_sda_layer++;

  while (num_sda_layer < result_num_layer) {
    answer = vector<vector<double>>(noisy_input);
    noisy_input = add_noise(denoisingAutoencoder.getMiddleOutput(noisy_input), 0.5);

    denoisingAutoencoder = DenoisingAutoencoder(noisy_input[0].size(), compression_rate,
                                                dropout_rate);
    sda_neurons[num_sda_layer] = denoisingAutoencoder.learn(answer, noisy_input);
    sda_out[num_sda_layer].resize(sda_neurons[num_sda_layer].size());

    num_sda_layer++;
  }
}

vector<double> StackedDenoisingAutoencoder::out(const vector<double> &input) {
  in = input;

  // Feed Forward
  // SdA First Layer
  unsigned long charge;
  threads.clear();
  if (sda_neurons[0].size() <= num_thread) charge = 1;
  else charge = sda_neurons[0].size() / num_thread;
  for (unsigned long i = 0, num_neuron = sda_neurons[0].size(); i < num_neuron; i += charge) {
    if (i != 0 && num_neuron / i == 1) {
      threads.push_back(thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                               i, num_neuron));
    } else {
      threads.push_back(thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                               i, i + charge));
    }
  }
  for (thread &th : threads) th.join();

  // SdA Other Layer
  if (sda_neurons.size() > 1) {
    for (unsigned long layer = 1, last_layer = sda_neurons.size() - 1;
         layer <= last_layer; ++layer) {
      threads.clear();
      if (sda_neurons[layer].size() <= num_thread) charge = 1;
      else charge = sda_neurons[layer].size() / num_thread;
      for (unsigned long i = 0, num_neuron = sda_neurons[layer].size();
           i < num_neuron; i += charge) {
        if (i != 0 && num_neuron / i == 1) {
          threads.push_back(thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                   layer, i, num_neuron));
        } else {
          threads.push_back(thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                   layer, i, i + charge));
        }
      }
      for (thread &th : threads) th.join();
    }
  }

  return sda_out.back();
}


void StackedDenoisingAutoencoder::sdaFirstLayerOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    sda_out[0][neuron] = sda_neurons[0][neuron].output(in);
}

void StackedDenoisingAutoencoder::sdaOtherLayerOutThread(const int layer,
                                                  const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    sda_out[layer][neuron] = sda_neurons[layer][neuron].output(sda_out[layer - 1]);
  }
}
