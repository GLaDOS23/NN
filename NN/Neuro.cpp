#include "neurocpp.h"
 std::vector <std::vector<std::vector<std::vector<double>>>> NeuroCPP::initialize_weights(int input_size, int hidden_neurons, int output_size, int  hidden_layers) {
     weights = std::vector < std::vector<std::vector<std::vector<double>>>>(1, std::vector < std::vector<std::vector<double>>>(0, std::vector < std::vector<double>>(0, std::vector <double>(0))));
     std::vector<std::vector<double>>z(hidden_neurons, std::vector<double>(input_size ));
     std::vector<std::vector<double>>z2(hidden_neurons, std::vector<double>(hidden_neurons));
     std::vector<std::vector<double>>z3(output_size, std::vector<double>(hidden_neurons ));
     std::vector<std::vector<double>>z4(input_size, std::vector<double>(output_size));
     // Создаем генератор случайных чисел с использованием текущего времени в качестве семени
     std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
     // Создаем распределение для чисел в диапазоне от 0 до 1 (включительно)
     std::uniform_real_distribution<double> dist(0.0, 1.0);
     if (hidden_layers > 0) {
         for (int i1 = 0; i1 < hidden_neurons; ++i1) {
             for (int i2 = 0; i2 < input_size ; ++i2) {
                 z[i1][i2] = dist(rng);
             }
         }
         weights[0].push_back(z);
         for (int i3 = 0; i3 < hidden_layers - 1; ++i3) {
             for (int i1 = 0; i1 < hidden_neurons; ++i1) {
                 for (int i2 = 0; i2 < hidden_neurons; ++i2) {
                     z2[i1][i2] = dist(rng);
                 }
             }
             weights[0].push_back(z2);
         }
         for (int i1 = 0; i1 < output_size; ++i1) {
             for (int i2 = 0; i2 < hidden_neurons ; ++i2) {
                 z3[i1][i2] = dist(rng);
             }
         }
         weights[0].push_back(z3);
     }
     else {
             for (int i1 = 0; i1 < input_size; ++i1) {
                 for (int i2 = 0; i2 < output_size; ++i2) {
                     z4[i1][i2] = dist(rng);
                 }
             }
             weights[0].push_back(z4);
     }
     return  weights;
 }
 std::vector < std::vector<std::vector<double>>> NeuroCPP::initialize_biases(int input_size, int hidden_neurons, int output_size, int  hidden_layers) {
     //std::vector<std::vector<double>>biases (xSize, std::vector<double>(ySize));
     biases = std::vector<std::vector<std::vector<double>>>(0, std::vector<std::vector<double>>(0, std::vector<double>(0)));
     std::vector<std::vector<double>>z(1, std::vector<double>(hidden_neurons));
     std::vector<std::vector<double>>z2(1, std::vector<double>(output_size));
     if (hidden_layers > 0) {
         biases.push_back(z);
         for (int i3 = 0; i3 < hidden_layers -1; ++i3) {

             biases.push_back(z);
         }
         biases.push_back(z2);
     }
     else {
         biases.push_back(z2);
     }
     return  biases;
 }
 //нейрон на входе 1- вектор входных значений, 2 - вектор biases, 3 - вектор weights
 std::vector<double> NeuroCPP::Neuron(const std::vector<double>& x, const std::vector<double>& biases, const std::vector<std::vector<double>>& weights) {
     return sigmoid_vector(VectorVectorAddition(biases, matrixVectorMultiply(x, weights)));
 }
 std::vector<double> NeuroCPP::FeedForward(const std::vector<double>& x) {
     std::vector<double> z;
     z = x;    
     for (int i = 0; i < biases.size(); i++) {
         z = Neuron(z, biases[i][0], weights[0][i]);
     }
     return z;
 }
 std::vector < std::vector<double>> NeuroCPP::FeedForward_activations(const std::vector<double>& x) {
     std::vector<double> z;
     z = x;
     activations = std::vector<std::vector<double>>(0, std::vector<double>(0));
     for (int i = 0; i < biases.size(); i++) {

         z = Neuron(z, biases[i][0], weights[0][i]);
         activations.push_back(z);

     }
     return activations;
 }
 //обратное распространение ошибки
  double NeuroCPP::Backpropagation(const std::vector<double>& x, const std::vector<double>& y,double learning_rate) {
     std::vector<double> Y;
     Y = y;
     FeedForward_activations(x);
     Y = VectorVectorProduct(VectorVectorSubtraction(Y, activations[biases.size() -1]), sigmoid_derivative_vector(activations[biases.size() -1]));
     weights[0][biases.size() - 1]= TransposeMatrix( matrixVectorAddition(VectorVectorProduct(activations[biases.size() - 1], VectorNumberProduct(Y, learning_rate)), TransposeMatrix(weights[0][biases.size() - 1])));
     biases[biases.size() - 1][0] =  VectorVectorAddition(biases[biases.size() - 1][0], VectorNumberProduct(Y, learning_rate));
     for (int i = biases.size()-2; i >=0 ; i--) {
        Y = VectorVectorProduct(matrixVectorMultiply(Y, TransposeMatrix(weights[0][i +1])), activations[i]);
        weights[0][ i ] = TransposeMatrix( matrixVectorAddition(VectorVectorProduct(activations[i ], VectorNumberProduct(Y, learning_rate)), TransposeMatrix(weights[0][ i ])));
        biases[ i ][0] = VectorVectorAddition(biases[ i ][0], VectorNumberProduct(Y, learning_rate));
     }
     return 0;
 }
  //запуск обучения известное количество раз (epochs)
  double NeuroCPP::Train(const std::vector<double>& x, const std::vector<double>& y, int epochs, double learning_rate) {
      for (int i = 0; i < epochs; i++) {
          Backpropagation(x, y, learning_rate);
      }
      return 0;
  }
  double  NeuroCPP::Print() {
      std::cout << "///////////////////////////////////////////////////////////" << std::endl;
      std::cout << "biases :" << std::endl;
      for (const std::vector < std::vector<double>>& row1 : biases) {
          for (const std::vector<double>& value1 : row1) {
              for (double value2 : value1) {
                  std::cout << value2 << " ";
              }
              std::cout << std::endl;
          }
          std::cout << std::endl;
      }
      std::cout << "weights :" << std::endl;
      
      for (const std::vector < std::vector < std::vector<double>>>& row1 : weights) {
          for (const std::vector < std::vector<double>>& value1 : row1) {
              for (std::vector <double> value2 : value1) {
                  for (double value3 : value2) {
                      std::cout << value3 << " ";
                  }
                  std::cout << std::endl;
              }
              std::cout << std::endl;
          }
          std::cout << std::endl;
      }
      std::cout<<"activations :" << std::endl;
      for (const  std::vector<double>& row1 : activations) {
          
              for (double value2 : row1) {
                  std::cout << value2 << " ";
              }
              std::cout << std::endl;
          
      }
      std::cout << std::endl;
      return 0;
  }
 //функция сигмоиды для векторов
 std::vector<double> NeuroCPP::sigmoid_vector(const std::vector<double>& x) {
     std::vector<double> result(x.size());
     for (size_t i = 0; i < x.size(); ++i) {
         result[i] = sigmoid(x[i]);
     }
     return result;
 }
 //функция производной сигмоиды для векторов
 std::vector<double> NeuroCPP::sigmoid_derivative_vector(const std::vector<double>& x) {
     std::vector<double> result(x.size());
     for (size_t i = 0; i < x.size(); ++i) {
         result[i] = sigmoid_derivative(x[i]);
     }
     return result;
 }
 double NeuroCPP::sigmoid(double x) {
     return(1 / (1 + exp(-x)));
 }
 double NeuroCPP::sigmoid_derivative(double x) {
     return(x * (1 - x));
 }