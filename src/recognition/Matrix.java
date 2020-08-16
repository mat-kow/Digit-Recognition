package recognition;

import java.util.Arrays;

public class Matrix {
    public static int determinant(int[][] matrix) {
        int size = matrix.length;
        if (size != matrix[0].length) throw new IllegalArgumentException();
        if (size == 2) return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
        int det = 0;
        for (int i = 0; i < size; i++) {
            int[][] subMatrix = new int [size - 1][size - 1];
            for (int k = 0; k < size - 1; k++) {
                for (int l = 0; l < size - 1; l++) {
                    int row = k < i ? k : k + 1;
                    subMatrix[k][l] = matrix[row][l + 1];
                }
            }
            det += Math.pow(-1, i)/*j = 0*/ * matrix[i][0] * determinant(subMatrix);
        }
        return det;
    }

    public static double determinant(double[][] matrix) {
        int size = matrix.length;
        if (size != matrix[0].length) throw new IllegalArgumentException();
        if (size == 2) return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
        double det = 0;
        for (int i = 0; i < size; i++) {
            double[][] subMatrix = new double[size - 1][size - 1];
            for (int k = 0; k < size - 1; k++) {
                for (int l = 0; l < size - 1; l++) {
                    int row = k < i ? k : k + 1;
                    subMatrix[k][l] = matrix[row][l + 1];
                }
            }
            det += Math.pow(-1, i)/*j = 0*/ * matrix[i][0] * determinant(subMatrix);
        }
        return det;
    }

    public static double[][] transposeMain(double[][] matrix) {
        double[][] result = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }
    public static int[][] transposeMain(int[][] matrix) {
        int[][] result = new int[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }

    public static double[][] multiplyMatrixByMatrix(double[][] matrix1, double[][] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int rows2 = matrix2.length;
        int cols2 = matrix2[0].length;
        if (cols1 != rows2) throw new IllegalArgumentException();
        double[][] result = new double[rows1][cols2];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                double n = 0;
                for (int k = 0; k < cols1; k++) {
                    n += matrix1[i][k] * matrix2[k][j];
                }
                result[i][j] = n;
            }
        }
        return result;
    }
    public static int[][] multiplyMatrixByMatrix(int[][] matrix1, int[][] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int rows2 = matrix2.length;
        int cols2 = matrix2[0].length;
        if (cols1 != rows2) throw new IllegalArgumentException();
        int[][] result = new int[rows1][cols2];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                int n = 0;
                for (int k = 0; k < cols1; k++) {
                    n += matrix1[i][k] * matrix2[k][j];
                }
                result[i][j] = n;
            }
        }
        return result;
    }
    public static int[][] multiplyMatrixByScalar(int[][] matrix, int multiplier) {
        int[][] result = Arrays.copyOf(matrix, matrix.length);
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] *= multiplier;
            }
        }
        return result;
    }
    public static double[][] multiplyMatrixByScalar(double[][] matrix, double multiplier) {
        double[][] result = Arrays.copyOf(matrix, matrix.length);
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] *= multiplier;
            }
        }
        return result;
    }

    public static int[][] addMatrix(int[][] matrix1, int[][] matrix2) {
        int[][] result = Arrays.copyOf(matrix1, matrix1.length);
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] += matrix2[i][j];
            }
        }
        return result;
    }
    public static double[][] addMatrix(double[][] matrix1, double[][] matrix2) {
        if (matrix1.length != matrix2.length) {
            throw new IllegalArgumentException("Matrices has different sizes(rows)");
        }
        if (matrix1[0].length != matrix2[0].length) {
            throw new IllegalArgumentException("Matrices has different sizes(columns)");
        }
//        double[][] result = Arrays.copyOf(matrix1, matrix1.length);
//        for (int i = 0; i < result.length; i++) {
//            for (int j = 0; j < result[i].length; j++) {
//                result[i][j] += matrix2[i][j];
//            }
//        }
        double[][] result = new double[matrix1.length][matrix1[0].length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] = matrix2[i][j] + matrix1[i][j];
            }
        }
        return result;
    }

    public static int[][] invertMatrix(int[][] matrix) {
        int det = determinant(matrix);
        int[][] cofactorsMatrix = cofactorsMatrix(matrix);
        int[][] transCofactorMatrix = transposeMain(cofactorsMatrix);
        return multiplyMatrixByScalar(transCofactorMatrix, 1 / det);
    }

    public static double[][] invertMatrix(double[][] matrix) {
        double det = determinant(matrix);
        double[][] cofactorsMatrix = cofactorsMatrix(matrix);
        double[][] transCofactorMatrix = transposeMain(cofactorsMatrix);
        return multiplyMatrixByScalar(transCofactorMatrix, 1 / det);
    }

    public static int[][] cofactorsMatrix(int[][] matrix) {
        int size = matrix.length;
        if (size != matrix[0].length) throw new IllegalArgumentException();
        int[][] result = new int[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int[][] subMatrix = new int[size - 1][size - 1];
                for (int k = 0; k < size - 1; k++) {
                    for (int l = 0; l < size - 1; l++) {
                        int subRow = k < i ? k : k + 1;
                        int subCol = l < j ? l : l + 1;
                        subMatrix[k][l] = matrix[subRow][subCol];
                    }
                }
                result[i][j] = (int) Math.pow(-1, i + j) * determinant(subMatrix);
            }
        }
        return result;
    }

    public static double[][] cofactorsMatrix(double[][] matrix) {
        int size = matrix.length;
        if (size != matrix[0].length) throw new IllegalArgumentException();
        double[][] result = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double[][] subMatrix = new double[size - 1][size - 1];
                for (int k = 0; k < size - 1; k++) {
                    for (int l = 0; l < size - 1; l++) {
                        int subRow = k < i ? k : k + 1;
                        int subCol = l < j ? l : l + 1;
                        subMatrix[k][l] = matrix[subRow][subCol];
                    }
                }
                result[i][j] = (int) Math.pow(-1, i + j) * determinant(subMatrix);
            }
        }
        return result;
    }
    public static double[][] unitMatrix(int size) {
        double[][] matrix = new double[size][size];
        for (int i = 0; i < size; i++) {
            matrix[i][i] = 1;
        }
        return matrix;
    }
}
