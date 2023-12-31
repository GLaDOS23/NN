#include "neurocpp.h"
//#include "Eigen/src/AccelerateSupport/AccelerateSupport.h"
// ������� ��� ��������� ������� �� �������

// ������� ��� �������� ���� ������
std::vector<std::vector<double>> NeuroCPP::addMatrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    int rows = matrix1.size();
    int cols = matrix1[0].size(); // ��������������, ��� ������� ����������� �������
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0)); // ������� �������-���������

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = matrix1[i][j] + matrix2[i][j]; // ���������� ��������������� ��������
        }
    }

    return result;
}


std::vector<double> NeuroCPP::matrixVectorMultiply(const std::vector<double>& vector, const std::vector<std::vector<double>>& matrix) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();


    std::vector<double> result(numRows, 0.0);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    //std::vector<double> result = employee.matrixVectorMultiply(matrix, vector)
    return result;
}

// ������� ��� �������� ������� � �������
std::vector<std::vector<double>> NeuroCPP::matrixVectorAddition(const std::vector<double>& vector, const std::vector<std::vector<double>>& matrix) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();



    std::vector<std::vector<double>> result(numRows, std::vector<double>(numCols, 0.0));

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            result[i][j] = matrix[i][j] + vector[j];
        }
    }
    //std::vector<std::vector<double>> result1 = employee.matrixVectorAddition(matrix, vector);
    return result;
}


//������� ��������� ������� �� �����
std::vector<double> NeuroCPP::VectorNumberProduct(const std::vector<double>& vector1, double x) {
    int numRows = vector1.size();
    std::vector<double> result(numRows);

    // �������� ��������� ��������
    for (size_t i = 0; i < numRows; ++i) {
        result[i] = vector1[i] * x;
    }

    return result;
}



//������� ��������� ���� ��������
std::vector<double> NeuroCPP::VectorVectorProduct(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    // ��������, ��� ��� ������� ����� ���������� ������
    int numRows = vector1.size();

    std::vector<double> result(numRows);

    // �������� ��������� ��������
    for (int i = 0; i < numRows; ++i) {
        result[i] = vector1[i] * vector2[i];
    }

    return result;
}


//������� �������� ���� ��������
std::vector<double> NeuroCPP::VectorVectorAddition(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    int numRows = vector1.size();
    std::vector<double> result(numRows);

    // �������� ��������� ��������
    for (size_t i = 0; i < numRows; ++i) {
        result[i] = vector1[i] + vector2[i];
    }

    return result;
}
//������� ��������� ���� ��������
std::vector<double> NeuroCPP::VectorVectorSubtraction(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    int numRows = vector1.size();
    std::vector<double> result(numRows);

    // ��������� ��������� ��������
    for (size_t i = 0; i < numRows; ++i) {
        result[i] = vector1[i] - vector2[i];
    }

    return result;
}
// ������� ��� ���������������� �������
std::vector<std::vector<double>> NeuroCPP::TransposeMatrix(const std::vector<std::vector<double>>& matrix) {
    // ������� ������� �������� �������
    int rows = matrix.size();
    int cols = matrix[0].size();

    // �������� ����� ������� � ����������� ���������
    std::vector<std::vector<double>> transposedMatrix(cols, std::vector<double>(rows));

    // �������� ����������������
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }

    return transposedMatrix;
}

double NeuroCPP::MSE_vector(const  std::vector<double>& x, const  std::vector<double>& y) {
    int numRows = x.size();
    double result = 0;
    for (size_t i = 0; i < numRows; ++i) {
        result += pow((x[i] - y[i]),2);
    }
    return  result / numRows ;
}
