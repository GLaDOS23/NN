#include "neurocpp.h"
//#include "Eigen/src/AccelerateSupport/AccelerateSupport.h"
// Функция для умножения вектора на матрицу
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

// Функция для сложения вектора и матрицы
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


//функция умножения вектора на число
std::vector<double> NeuroCPP::VectorNumberProduct(const std::vector<double>& vector1, double x) {
    int numRows = vector1.size();
    std::vector<double> result(numRows);

    // Сложение элементов векторов
    for (size_t i = 0; i < numRows; ++i) {
        result[i] = vector1[i] * x;
    }

    return result;
}



//функция умножения двух векторов
std::vector<double> NeuroCPP::VectorVectorProduct(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    // Убедимся, что оба вектора имеют одинаковый размер
    int numRows = vector1.size();

    std::vector<double> result(numRows);

    // Сложение элементов векторов
    for (int i = 0; i < numRows; ++i) {
        result[i] = vector1[i] * vector2[i];
    }

    return result;
}


//функция сложения двух векторов
std::vector<double> NeuroCPP::VectorVectorAddition(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    int numRows = vector1.size();
    std::vector<double> result(numRows);

    // Сложение элементов векторов
    for (size_t i = 0; i < numRows; ++i) {
        result[i] = vector1[i] + vector2[i];
    }

    return result;
}
//функция вычитания двух векторов
std::vector<double> NeuroCPP::VectorVectorSubtraction(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    int numRows = vector1.size();
    std::vector<double> result(numRows);

    // Сложение элементов векторов
    for (size_t i = 0; i < numRows; ++i) {
        result[i] = vector1[i] - vector2[i];
    }

    return result;
}
// Функция для транспонирования матрицы
std::vector<std::vector<double>> NeuroCPP::TransposeMatrix(const std::vector<std::vector<double>>& matrix) {
    // Получим размеры исходной матрицы
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Создадим новую матрицу с обмененными размерами
    std::vector<std::vector<double>> transposedMatrix(cols, std::vector<double>(rows));

    // Выполним транспонирование
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }

    return transposedMatrix;
}