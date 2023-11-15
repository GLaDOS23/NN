#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>

class NeuroCPP {
public:

    typedef std::vector<double> v1;
    typedef std::vector<v1> v2;
    typedef std::vector<v2> v3;
    typedef std::vector<v3> v4;
    v2 activations;
    v2 biases;
    v3 weights;


    // Функция для сложения двух матриц
    std::vector<std::vector<double>> addMatrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2);
    // Функция для умножения вектора на матрицу
     std::vector<double> matrixVectorMultiply(const std::vector<double>& vector, const std::vector<std::vector<double>>& matrix);

    // Функция для сложения вектора и матрицы
     std::vector<std::vector<double>> matrixVectorAddition(const std::vector<double>& vector, const std::vector<std::vector<double>>& matrix);

     //функция умножения вектора на число
     std::vector<double> VectorNumberProduct(const std::vector<double>& vector1, double x);
     std::vector<double> VectorVectorProduct(const std::vector<double>& vector1, const std::vector<double>& vector2);

    //функция сложения двух векторов
     std::vector<double> VectorVectorAddition(const std::vector<double>& vector1, const std::vector<double>& vector2);
     std::vector<double> VectorVectorSubtraction(const std::vector<double>& vector1, const std::vector<double>& vector2);
     std::vector<std::vector<double>> TransposeMatrix(const std::vector<std::vector<double>>& matrix);
   
     std::vector<std::vector<std::vector<double>>> initialize_weights(int input_size, int hidden_neurons, int output_size, int  hidden_layers);

    
     std::vector<std::vector<double>> initialize_biases(int input_size, int hidden_neurons, int output_size, int  hidden_layers);


    //нейрон на входе 1- вектор входных значений, 2 - вектор biases, 3 - вектор weights
     std::vector<double> Neuron(const std::vector<double>& x, const std::vector<double>& biases, const std::vector<std::vector<double>>& weights);
     std::vector < std::vector<double>>FeedForward_activations(const std::vector<double>& x);//, const std::vector<std::vector<double>>& weights
    std::vector<double> FeedForward(const std::vector<double>& x);//, const std::vector<std::vector<double>>& weights
    double Backpropagation(const std::vector<double>& x, const std::vector<double>& y,double learning_rate);
    double Backpropagation2(const std::vector<double>& x, const std::vector<double>& y, double learning_rate);
    double Train(const std::vector < std::vector<double>>& x, const std::vector < std::vector<double>>& y, int epochs, double learning_rate);
    double MSE_vector(const  std::vector<double>& x, const std::vector<double>& y);
    //функция сигмоиды для векторов

     std::vector<double> sigmoid_vector(const std::vector<double>& x);
    //функция производной сигмоиды для векторов
     std::vector<double> sigmoid_derivative_vector(const std::vector<double>& x);
     double sigmoid(double x);
     double sigmoid_derivative(double x);
     double  Print();

private:

};
