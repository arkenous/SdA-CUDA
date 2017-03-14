
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
  sda_learned_out.resize(result_num_layer);
  unsigned long num_sda_layer = 0;

  vector<vector<double>> answer(input);
  vector<vector<double>> noisy_input(gaussian_noise(input, 0.0, 1.0, noised_rate));

  DenoisingAutoencoder denoisingAutoencoder(noisy_input[0].size(), compression_rate, dropout_rate);

  // Store learned dA middle neurons
  sda_neurons[num_sda_layer] = denoisingAutoencoder.learn(answer, noisy_input);
  sda_out[num_sda_layer].resize(sda_neurons[num_sda_layer].size());
  sda_learned_out[num_sda_layer].resize(sda_neurons[num_sda_layer].size());

  num_sda_layer++;

  while (num_sda_layer < result_num_layer) {
    answer = vector<vector<double>>(noisy_input);
    noisy_input = gaussian_noise(denoisingAutoencoder.getMiddleOutput(noisy_input), 0.0, 1.0, noised_rate);

    denoisingAutoencoder = DenoisingAutoencoder(noisy_input[0].size(), compression_rate,
                                                dropout_rate);
    sda_neurons[num_sda_layer] = denoisingAutoencoder.learn(answer, noisy_input);
    sda_out[num_sda_layer].resize(sda_neurons[num_sda_layer].size());
    sda_learned_out[num_sda_layer].resize(sda_neurons[num_sda_layer].size());

    num_sda_layer++;
  }
}

void StackedDenoisingAutoencoder::learn(const vector <vector<double>> &input,
                                        const vector <vector<double>> &answer,
                                        const double dropout_rate) {
  vector<double> empty_vector;
  output_neuron = Neuron(sda_neurons.back().size(), empty_vector, empty_vector, empty_vector,
                         0, 0.0, 1, 0.0);
  output_neuron.dropout(1.0); // disable dropout

  // Learn
  int succeed = 0; // 連続正解回数のカウンタを初期化

  random_device rnd;
  mt19937 mt;
  mt.seed(rnd());
  uniform_real_distribution<double> real_rnd(0.0, 1.0);
  unsigned long layer = 0, neuron = 0, n_size = 0, i = 0, j = 0;
  unsigned long sda_neuron_size = sda_neurons.size();
  unsigned long answer_size = answer.size();
  unsigned long charge = 0;
  unsigned long input_size = input.size();

  for (int trial = 0; trial < MAX_TRIAL; ++trial) {
    cout << "-----   trial: " << trial << "   -----" << endl;

    // Set SdA dropout
    for (layer = 0; layer < sda_neuron_size; ++layer) {
      for (neuron = 0, n_size = sda_neurons[layer].size(); neuron < n_size; ++neuron) {
        sda_neurons[layer][neuron].dropout(real_rnd(mt));
      }
    }

    in = input[trial % answer_size];
    ans = answer[trial % answer_size];

    // Feed Forward
    // SdA First Layer
    threads.clear();
    if (sda_neurons[0].size() <= num_thread) {
      for (i = 0, n_size = sda_neurons[0].size(); i < n_size; ++i)
        threads[i] = thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                                    i, i + 1);
      for (i = 0, n_size = sda_neurons[0].size(); i < n_size; ++i)
        threads[i].join();
    } else {
      charge = sda_neurons[0].size() / num_thread;
      for (i = 0, j = 0, n_size = sda_neurons[0].size(); j < num_thread; i += charge, ++j)
        if (j == num_thread - 1)
          threads[j] = thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                                   i, n_size);
        else
          threads[j] = thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                                   i, i + charge);
      for (i = 0, j = 0, n_size = sda_neurons[0].size(); j < num_thread; i += charge, ++j)
        threads[j].join();
    }

    // SdA Other Layer
    if (sda_neuron_size > 1) {
      for (layer = 1; layer < sda_neuron_size; ++layer) {
        threads.clear();
        if (sda_neurons[layer].size() <= num_thread) {
          for (i = 0, n_size = sda_neurons[layer].size(); i < n_size; ++i)
            threads[i] = thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                      layer, i, i + 1);
          for (i = 0, n_size = sda_neurons[layer].size(); i < n_size; ++i)
            threads[i].join();
        } else {
          charge = sda_neurons[layer].size() / num_thread;
          for (i = 0, j = 0, n_size = sda_neurons[layer].size(); j < num_thread; i += charge, ++j)
            if (j == num_thread - 1)
              threads[j] = thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                       layer, i, n_size);
            else
              threads[j] = thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                       layer, i, i + charge);
          for (i = 0, j = 0, n_size = sda_neurons[layer].size(); j < num_thread; i += charge, ++j)
            threads[j].join();
        }
      }
    }

    // 出力値を推定
    threads.clear();
    if (output_neuron_num <= num_thread) {
      for (i = 0; i < output_neuron_num; ++i)
        threads[i] = thread(&StackedDenoisingAutoencoder::outForwardThread, this,
                                    i, i + 1);
      for (i = 0; i < output_neuron_num; ++i)
        threads[i].join();
    } else {
      charge = output_neuron_num / num_thread;
      for (i = 0, j = 0; j < num_thread; i += charge, ++j)
        if (j == num_thread - 1)
          threads[j] = thread(&StackedDenoisingAutoencoder::outForwardThread, this,
                                   i, output_neuron_num);
        else
          threads[j] = thread(&StackedDenoisingAutoencoder::outForwardThread, this,
                                   i, i + charge);
      for (i = 0, j = 0; j < num_thread; i += charge, ++j)
        threads[j].join();
    }

    successFlg = true;

    // Back Propagation (learn phase)
    //region 出力層を学習する
    threads.clear();
    if (output_neuron_num <= num_thread) {
      for (i = 0; i < output_neuron_num; ++i)
        threads[i] = thread(&StackedDenoisingAutoencoder::outLearnThread, this,
                                    i, i + 1);
      for (i = 0; i < output_neuron_num; ++i)
        threads[i].join();
    } else {
      charge = output_neuron_num / num_thread;
      for (i = 0, j = 0; j < num_thread; i += charge, ++j)
        if (j == num_thread - 1)
          threads[j] = thread(&StackedDenoisingAutoencoder::outLearnThread, this,
                                   i, output_neuron_num);
        else
          threads[j] = thread(&StackedDenoisingAutoencoder::outLearnThread, this,
                                   i, i + charge);
      for (i = 0, j = 0; j < num_thread; i += charge, ++j)
        threads[j].join();
    }
    //endregion

    // 連続成功回数による終了判定
    if (successFlg) {
      ++succeed;
      if (succeed >= input_size) break;
      else continue;
    } else succeed = 0;

    // learn SdA
//    if (sda_neurons.size() > 1) {
//      threads.clear();
//      if (sda_neurons[sda_neurons.size() - 1].size() <= num_thread) charge = 1;
//      else charge = sda_neurons[sda_neurons.size() - 1].size() / num_thread;
//      for (int i = 0; i < sda_neurons[sda_neurons.size() - 1].size(); i += charge) {
//        if (i != 0 && sda_neurons[sda_neurons.size() - 1].size() / i == 1) {
//          threads.push_back(std::thread(&StackedDenoisingAutoencoder::sdaLastLayerLearnThread, this,
//                                        i, sda_neurons[sda_neurons.size() - 1].size()));
//        } else {
//          threads.push_back(std::thread(&StackedDenoisingAutoencoder::sdaLastLayerLearnThread, this,
//                                        i, i + charge));
//        }
//      }
//      for (std::thread &th : threads) th.join();
//    }
//
//    for (int layer = sda_neurons.size() - 2; layer >= 1; --layer) {
//      if (sda_neurons[layer].size() <= num_thread) charge = 1;
//      else charge = sda_neurons[layer].size() / num_thread;
//      threads.clear();
//      for (int i = 0; i < sda_neurons[layer].size(); i += charge) {
//        if (i != 0 && sda_neurons[layer].size() / i == 1) {
//          threads.push_back(std::thread(&StackedDenoisingAutoencoder::sdaMiddleLayerLearnThread, this,
//                                        layer, i, sda_neurons[layer].size()));
//        } else {
//          threads.push_back(std::thread(&StackedDenoisingAutoencoder::sdaMiddleLayerLearnThread, this,
//                                        layer, i, i + charge));
//        }
//      }
//      for (std::thread &th : threads) th.join();
//    }
//
//    threads.clear();
//    if (sda_neurons[0].size() <= num_thread) charge = 1;
//    else charge = sda_neurons[0].size() / num_thread;
//    for (int i = 0; i < sda_neurons[0].size(); i += charge) {
//      if (i != 0 && sda_neurons[0].size() / i == 1) {
//        threads.push_back(std::thread(&StackedDenoisingAutoencoder::sdaFirstLayerLearnThread, this,
//                                      i, sda_neurons[0].size()));
//      } else {
//        threads.push_back(std::thread(&StackedDenoisingAutoencoder::sdaFirstLayerLearnThread, this,
//                                      i, i + charge));
//      }
//    }
//    for (std::thread &th : threads) th.join();
  }

  // 全ての教師データで正解を出すか，学習上限回数を超えた場合に終了
}


void StackedDenoisingAutoencoder::sdaFirstLayerOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    sda_learned_out[0][neuron] = sda_neurons[0][neuron].output(in);
}

void StackedDenoisingAutoencoder::sdaOtherLayerOutThread(const int layer,
                                                         const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    sda_learned_out[layer][neuron] = sda_neurons[layer][neuron].output(sda_learned_out[layer - 1]);
  }
}

void StackedDenoisingAutoencoder::outForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    o = output_neuron.learn_output(sda_learned_out.back());
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
    output_neuron.learn(delta, sda_learned_out.back());
  }
}

double StackedDenoisingAutoencoder::crossEntropy(const double output, const double answer) {
  return -answer * log(output) - (1.0 - answer) * log(1.0 - output);
}


double StackedDenoisingAutoencoder::out(const vector<double> &input) {
  in = input;

  unsigned long n_size = 0, charge = 0, i = 0, j = 0, layer = 0;
  unsigned long sda_neuron_size = sda_neurons.size();

  threads.clear();
  if (sda_neurons[0].size() <= num_thread) {
    for (i = 0, n_size = sda_neurons[0].size(); i < n_size; ++i)
      threads[i] = thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                                  i, i + 1);
    for (i = 0, n_size = sda_neurons[0].size(); i < n_size; ++i)
      threads[i].join();
  } else {
    charge = sda_neurons[0].size() / num_thread;
    for (i = 0, j = 0, n_size = sda_neurons[0].size(); j < num_thread; i += charge, ++j)
      if (j == num_thread - 1)
        threads[j] = thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                                 i, n_size);
      else
        threads[j] = thread(&StackedDenoisingAutoencoder::sdaFirstLayerOutThread, this,
                                 i, i + charge);
    for (i = 0, j = 0, n_size = sda_neurons[0].size(); j < num_thread; i += charge, ++j)
      threads[j].join();
  }

  // SdA Other Layer
  if (sda_neurons.size() > 1) {
    for (layer = 1; layer < sda_neuron_size; ++layer) {
      threads.clear();
      if (sda_neurons[layer].size() <= num_thread) {
        for (i = 0, n_size = sda_neurons[layer].size(); i < n_size; ++i)
          threads[i] = thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                      layer, i, i + 1);
        for (i = 0, n_size = sda_neurons[layer].size(); i < n_size; ++i)
          threads[i].join();
      } else {
        charge = sda_neurons[layer].size() / num_thread;
        for (i = 0, j = 0, n_size = sda_neurons[layer].size(); j < num_thread; i += charge, ++j)
          if (j == num_thread - 1)
            threads[j] = thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                     layer, i, n_size);
          else
            threads[j] = thread(&StackedDenoisingAutoencoder::sdaOtherLayerOutThread, this,
                                     layer, i, i + charge);
        for (i = 0, j = 0, n_size = sda_neurons[layer].size(); j < num_thread; i += charge, ++j)
          threads[j].join();
      }
    }
  }

  // 出力値を推定
  threads.clear();
  if (output_neuron_num <= num_thread) {
    for (i = 0; i < output_neuron_num; ++i)
      threads[i] = thread(&StackedDenoisingAutoencoder::outOutThread, this,
                                  i, i + 1);
    for (i = 0, j = 0; i < output_neuron_num; ++i, ++j)
      threads[i].join();
  } else {
    charge = output_neuron_num / num_thread;
    for (i = 0, j = 0; j < num_thread; i += charge, ++j)
      if (j == num_thread - 1)
        threads[j] = thread(&StackedDenoisingAutoencoder::outOutThread, this,
                                 i, output_neuron_num);
      else
        threads[j] = thread(&StackedDenoisingAutoencoder::outOutThread, this,
                                 i, i + charge);
    for (i = 0, j = 0; j < num_thread; i += charge, ++j)
      threads[j].join();
  }


  return learned_o;
}


void StackedDenoisingAutoencoder::outOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    learned_o = output_neuron.output(sda_learned_out.back());
  }
}


/*
void StackedDenoisingAutoencoder::sdaFirstLayerForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    sda_out[0][neuron] = sda_neurons[0][neuron].learn_output(in);
}

void StackedDenoisingAutoencoder::sdaOtherLayerForwardThread(const int layer,
                                                             const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    sda_out[layer][neuron] = sda_neurons[layer][neuron].learn_output(sda_out[layer - 1]);
}

void StackedDenoisingAutoencoder::sdaLastLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;
    sumDelta = output_neuron.getInputWeightIndexOf(neuron) * output_neuron.getDelta();

    double delta;
    // sigmoid
    delta = (sda_out[sda_out.size() - 1][neuron]
             * (1.0 - sda_out[sda_out.size() - 1][neuron])) * sumDelta;

    sda_neurons[sda_neurons.size() - 1][neuron].learn(delta, sda_out[sda_out.size() - 2]);
  }
}

void StackedDenoisingAutoencoder::sdaMiddleLayerLearnThread(const int layer,
                                                            const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;
    for (int k = 0; k < sda_neurons[layer + 1].size(); ++k) {
      sumDelta += sda_neurons[layer + 1][k].getInputWeightIndexOf(neuron)
                  * sda_neurons[layer + 1][k].getDelta();
    }

    double delta;
    // sigmoid
    delta = (sda_out[layer][neuron] * (1.0 - sda_out[layer][neuron])) * sumDelta;

    sda_neurons[layer][neuron].learn(delta, sda_out[layer - 1]);
  }
}

void StackedDenoisingAutoencoder::sdaFirstLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;

    if (sda_neurons.size() > 1) {
      for (int k = 0; k < sda_neurons[1].size(); ++k) {
        sumDelta += sda_neurons[1][k].getInputWeightIndexOf(neuron) * sda_neurons[1][k].getDelta();
      }
    } else {
      sumDelta = output_neuron.getInputWeightIndexOf(neuron) * output_neuron.getDelta();
    }

    double delta;
    // sigmoid
    delta = (sda_out[0][neuron] * (1.0 - sda_out[0][neuron])) * sumDelta;

    sda_neurons[0][neuron].learn(delta, in);
  }
}
*/
