#include <iostream>
#include <vector>
#include <cmath>
#include "neurocpp.h"// � neurocpp.cpp ������� �� .h
#include <random>
#include <ctime>// �������� srand() � rand()

using namespace std;
int main() {
    NeuroCPP employee;
    std::vector<std::vector<double>> matrix = { {2.0, 2.0},
                                               {4.0, 5.0},
                                               {7.0, 8.0} };

    std::vector<double> vector = { 2.0, 3.0, 4.0 };
    std::vector<std::vector<double>> vector1 = { {1,2 },{2,0}};
    std::vector<std::vector<double>> vector2 = { {1},{0} };

    int input_size = 2;
    int output_size = 1;
    int hidden_layers = 2;
    int hidden_neurons = 5;
    int epochs = 1000;
    double learning_rate = 0.5;

    int xSize = 3;
    int ySize = 2;
    int zSize = 3;
    int iSize = 3;
    //std::vector<std::vector<double>>biases(xSize, std::vector<double>(ySize));
    //std::vector < std::vector<double>>biases2(2,std::vector<double>(20));
   //biases.insert(std::end(biases), std::begin(biases2), std::end(biases2));
    //std::vector<std::vector<std::vector<double>>>biases = 
    employee.initialize_biases(input_size, hidden_neurons, output_size, hidden_layers);
    //(xSize, std::vector < std::vector<std::vector<double>>>(ySize, std::vector < std::vector<double>>(zSize, std::vector <double>( iSize))))
    //std::vector < std::vector<std::vector<std::vector<double>>>> weights;
    //std::vector < std::vector<std::vector<std::vector<double>>>> weights = 
    employee.initialize_weights(input_size, hidden_neurons, output_size, hidden_layers);
    
    //employee.Print();

    /*
    std::vector < std::vector<double>>result3 = employee.FeedForward_activations(vector1);
    for (const  std::vector<double>& row1 : result3) {
        for (double value1 : row1) {
 
                std::cout << value1 << " ";

        }
        std::cout << std::endl;
    }
    std::cout << std::endl;*/
    
     //employee.Backpropagation(vector1,vector2, learning_rate);
    employee.Print();
    unsigned int start_time = clock();
    employee.Train(vector1, vector2, epochs, learning_rate);
    unsigned int end_time = clock();
    cout << "Time: " << (end_time - start_time)/ 1000.0 << endl;
        employee.Print();
        
    /*
    std::vector<double> result3 = employee.FeedForward(vector1, biases, weights);
    for (double row1 : result3) {
    
      std::cout << row1 << " ";
    }
    std::cout << std::endl;
    //std::cout << employee.sigmoid_derivative(8)<< endl;
   // std::vector<double> result = employee.matrixVectorMultiply(vector, matrix);
    //std::vector<std::vector<double>> result1 = employee.matrixVectorAddition( vector, matrix);
    //std::vector<double> result2 = employee.VectorVectorAddition(vector1, vector2);
   // std::vector<double> result3 = employee.Neuron(vector1, vector2, matrix);
    // ����� ����������
   
        for (double value : result3) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        */

    /*
    std::cout << std::endl;

    for (double value : result) {
        std::cout << value << " " ;
    }
    std::cout << std::endl;
    for (const std::vector<double>& row : result1) {
        for (double value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (const int& value : result2) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    for (const int& value : result3) {
        std::cout << value << " ";
    }*/
    return 0;
}