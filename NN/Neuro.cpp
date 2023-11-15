#include "neurocpp.h"
std::vector<std::vector<std::vector<double>>> NeuroCPP::initialize_weights(int input_size, int hidden_neurons, int output_size, int  hidden_layers) {
    weights = std::vector<std::vector<std::vector<double>>>(0, std::vector<std::vector<double>>(0, std::vector<double>(0)));
    std::vector<std::vector<double>>z(hidden_neurons, std::vector<double>(input_size));
    std::vector<std::vector<double>>z2(hidden_neurons, std::vector<double>(hidden_neurons));
    std::vector<std::vector<double>>z3(output_size, std::vector<double>(hidden_neurons));
    std::vector<std::vector<double>>z4(input_size, std::vector<double>(output_size));

    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
    // ������� ������������� ��� ����� � ��������� �� 0 �� 1 (������������)
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (hidden_layers > 0) {
        for (int i1 = 0; i1 < hidden_neurons; ++i1) {
            for (int i2 = 0; i2 < input_size; ++i2) {
                z[i1][i2] = dist(rng);
            }
        }
        weights.push_back(z);
        for (int i3 = 0; i3 < hidden_layers - 1; ++i3) {
            for (int i1 = 0; i1 < hidden_neurons; ++i1) {
                for (int i2 = 0; i2 < hidden_neurons; ++i2) {
                    z2[i1][i2] = dist(rng);
                }
            }
            weights.push_back(z2);
        }
        for (int i1 = 0; i1 < output_size; ++i1) {
            for (int i2 = 0; i2 < hidden_neurons; ++i2) {
                z3[i1][i2] = dist(rng);
            }
        }
        weights.push_back(z3);
    }
    else {
        for (int i1 = 0; i1 < input_size; ++i1) {
            for (int i2 = 0; i2 < output_size; ++i2) {
                z4[i1][i2] = dist(rng);
            }
        }
        weights.push_back(z4);
    }

    return  weights;
}
std::vector<std::vector<double>> NeuroCPP::initialize_biases(int input_size, int hidden_neurons, int output_size, int  hidden_layers) {
    //std::vector<std::vector<double>>biases (xSize, std::vector<double>(ySize));
    biases = std::vector<std::vector<double>>(0, std::vector<double>(0));
    std::vector<double>z(hidden_neurons);
    std::vector<double>z2(output_size);
    if (hidden_layers > 0) {
        biases.push_back(z);
        for (int i3 = 0; i3 < hidden_layers - 1; ++i3) {

            biases.push_back(z);
        }
        biases.push_back(z2);
    }
    else {
        biases.push_back(z2);
    }


    return  biases;


}
//������ �� ����� 1- ������ ������� ��������, 2 - ������ biases, 3 - ������ weights
std::vector<double> NeuroCPP::Neuron(const std::vector<double>& x, const std::vector<double>& biases, const std::vector<std::vector<double>>& weights) {
    return sigmoid_vector(VectorVectorAddition(biases, matrixVectorMultiply(x, weights)));
}
std::vector<double> NeuroCPP::FeedForward(const std::vector<double>& x) {
    std::vector<double> z;
    z = x;
    for (int i = 0; i < biases.size(); i++) {
        z = Neuron(z, biases[i], weights[i]);
    }
    return z;
}
std::vector < std::vector<double>> NeuroCPP::FeedForward_activations(const std::vector<double>& x) {
    std::vector<double> z;
    z = x;
    activations = std::vector<std::vector<double>>(0, std::vector<double>(0));
    for (int i = 0; i < biases.size(); i++) {

        z = Neuron(z, biases[i], weights[i]);
        activations.push_back(z);

    }
    return activations;
}
//�������� ��������������� ������
double NeuroCPP::Backpropagation(const std::vector<double>& x, const std::vector<double>& y, double learning_rate) {
    std::vector<double> Y;
    Y = y;
    FeedForward_activations(x);
    Y = VectorVectorProduct(VectorVectorSubtraction(Y, activations[biases.size() - 1]), sigmoid_derivative_vector(activations[biases.size() - 1]));
    weights[biases.size() - 1] = addMatrices(weights[biases.size() - 1], TransposeMatrix(matrixVectorAddition(VectorVectorProduct(activations[biases.size() -1], VectorNumberProduct(Y, learning_rate)), TransposeMatrix(weights[biases.size() - 1]))));
    biases[biases.size() - 1] = VectorVectorAddition(biases[biases.size() - 1], VectorNumberProduct(Y, learning_rate));
    for (int i = biases.size() - 2; i >= 0; i--) {
        Y = VectorVectorProduct(matrixVectorMultiply(Y, TransposeMatrix(weights[i + 1])), activations[i]);
        weights[i] = addMatrices(weights[i], TransposeMatrix(matrixVectorAddition(VectorVectorProduct(activations[i], VectorNumberProduct(Y, learning_rate)), TransposeMatrix(weights[i]))));
        biases[i] = VectorVectorAddition(biases[i], VectorNumberProduct(Y, learning_rate));
    }
    return 0; 
}
double NeuroCPP::Backpropagation2(const std::vector<double>& x, const std::vector<double>& y, double learning_rate) {
    std::vector<double> Y;
    Y = y;
    FeedForward_activations(x);
    Y = VectorVectorProduct(VectorVectorSubtraction(Y, activations[biases.size() - 1]), sigmoid_derivative_vector(activations[biases.size() - 1]));
    weights[biases.size() - 1] = TransposeMatrix(matrixVectorAddition(VectorVectorProduct(activations[biases.size() - 1], VectorNumberProduct(Y, learning_rate)), TransposeMatrix(weights[biases.size() - 1])));
    biases[biases.size() - 1] = VectorVectorAddition(biases[biases.size() - 1], VectorNumberProduct(Y, learning_rate));
    for (int i = biases.size() - 2; i >= 0; i--) {
        Y = VectorVectorProduct(matrixVectorMultiply(Y, TransposeMatrix(weights[i + 1])), activations[i]);
        weights[i] = TransposeMatrix(matrixVectorAddition(VectorVectorProduct(activations[i], VectorNumberProduct(Y, learning_rate)), TransposeMatrix(weights[i])));
        biases[i] = VectorVectorAddition(biases[i], VectorNumberProduct(Y, learning_rate));
    }
    return 0;
}
//������ �������� ��������� ���������� ��� (epochs)
//��� ��������, ������)
double NeuroCPP::Train(const  std::vector<std::vector<double>>& x, const  std::vector<std::vector<double>>& y, int epochs, double learning_rate) {
    int it = 0;
    for (int i = 0; i < epochs; i++) {
        for (int i2 = 0; i2 < y.size(); i2++) {

            Backpropagation(x[i2], y[i2], learning_rate);
        }

    }
    return 0;
}
double  NeuroCPP::Print() {
    std::cout << "///////////////////////////////////////////////////////////" << std::endl;
    std::cout << "biases :" << std::endl;
    
        for (const std::vector<double>& value1 : biases) {
            for (double value2 : value1) {
                std::cout << value2 << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    
    std::cout << "weights :" << std::endl;

   
        for (const std::vector < std::vector<double>>& value1 : weights) {
            for (std::vector <double> value2 : value1) {
                for (double value3 : value2) {
                    std::cout << value3 << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    
    std::cout << "activations :" << std::endl;
    for (const std::vector<double>& row1 : activations) {

        for (double value2 : row1) {
            std::cout << value2 << " ";
        }
        std::cout << std::endl;

    }
    std::cout << std::endl;

    std::cout <<"dddddddddddddd"<< std::endl;
        for (double value : activations[biases.size() -1]) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    
    return 0;
}
//������� �������� ��� ��������
std::vector<double> NeuroCPP::sigmoid_vector(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}
//������� ����������� �������� ��� ��������
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