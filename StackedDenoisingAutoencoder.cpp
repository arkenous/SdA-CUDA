
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
using std::cout;
using std::endl;

StackedDenoisingAutoencoder::StackedDenoisingAutoencoder() {
  threads = vector<thread>(num_thread);
}


void StackedDenoisingAutoencoder::build(const vector<vector<double>> &input,
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

void StackedDenoisingAutoencoder::learn(const vector <vector<double>> &input,
                                        const vector <vector<double>> &answer,
                                        const double dropout_rate) {
  vector<double> empty_vector;
  output_neuron = Neuron(sda_neurons.back().size(), empty_vector, empty_vector, empty_vector,
                         0, 0.0, 1, dropout_rate);
  output_neuron.dropout(1.0); // disable dropout

  // Learn
  int succeed = 0; // 連続正解回数のカウンタを初期化

  for (int trial = 0; trial < MAX_TRIAL; ++trial) {
    cout << "-----   trial: " << trial << "   -----" << endl;

    in = input[trial % answer.size()];
    ans = answer[trial % answer.size()];

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

    // 出力値を推定
    threads.clear();
    if (output_neuron_num <= num_thread) charge = 1;
    else charge = output_neuron_num / num_thread;
    for (int i = 0; i < output_neuron_num; i += charge) {
      if (i != 0 && output_neuron_num / i == 1) {
        threads.push_back(thread(&StackedDenoisingAutoencoder::outOutThread, this,
                                 i, output_neuron_num));
      } else {
        threads.push_back(thread(&StackedDenoisingAutoencoder::outOutThread, this,
                                 i, i + charge));
      }
    }
    for (thread &th : threads) th.join();

    successFlg = true;

    // Back Propagation (learn phase)
    //region 出力層を学習する
    threads.clear();
    if (output_neuron_num <= num_thread) charge = 1;
    else charge = output_neuron_num / num_thread;
    for (int i = 0; i < output_neuron_num; i += charge) {
      if (i != 0 && output_neuron_num / i == 1) {
        threads.push_back(thread(&StackedDenoisingAutoencoder::outLearnThread, this,
                                 i, output_neuron_num));
      } else {
        threads.push_back(thread(&StackedDenoisingAutoencoder::outLearnThread, this,
                                 i, i + charge));
      }
    }
    for (thread &th : threads) th.join();
    //endregion

    // 連続成功回数による終了判定
    if (successFlg) {
      succeed++;
      if (succeed >= input.size()) break;
      else continue;
    } else succeed = 0;

    //TODO SdAの学習も追加してもいいかも
  }

  // 全ての教師データで正解を出すか，学習上限回数を超えた場合に終了
}


double StackedDenoisingAutoencoder::out(const vector<double> &input) {
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

  // 出力値を推定
  threads.clear();
  if (output_neuron_num <= num_thread) charge = 1;
  else charge = output_neuron_num / num_thread;
  for (int i = 0; i < output_neuron_num; i += charge) {
    if (i != 0 && output_neuron_num / i == 1) {
      threads.push_back(thread(&StackedDenoisingAutoencoder::outOutThread, this,
                               i, output_neuron_num));
    } else {
      threads.push_back(thread(&StackedDenoisingAutoencoder::outOutThread, this,
                               i, i + charge));
    }
  }
  for (thread &th : threads) th.join();


  return o;
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

void StackedDenoisingAutoencoder::outOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    o = output_neuron.output(sda_out.back());
  }
}

void StackedDenoisingAutoencoder::outLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 出力層ニューロンのdeltaの計算
    double delta = o - ans[neuron];

    // 教師データとの誤差が十分小さい場合は学習しない．そうでなければ正解フラグをfalseに
    if (crossEntropy(o, ans[neuron]) < MAX_GAP) continue;
    else {
      cout << "MLP ce: " << crossEntropy(o, ans[neuron]) << endl;
      successFlg = false;
    }

    // 出力層の学習
    output_neuron.learn(delta, sda_out.back());
  }
}


double StackedDenoisingAutoencoder::crossEntropy(const double output, const double answer) {
  return -answer * log(output) - (1.0 - answer) * log(1.0 - output);
}

