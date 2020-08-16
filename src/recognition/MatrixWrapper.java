package recognition;

import java.io.IOException;
import java.io.Serializable;

public class MatrixWrapper implements Serializable {
    private static final long serialVersionUID = 1L;

    private double[][] matrix;

    private MatrixWrapper(double[][] matrix) {
        this.matrix = matrix;
    }

    public static void save(double[][] matrix, String fileName) {
        try {
            SerializationUtils.serialize(new MatrixWrapper(matrix), fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double[][] load(String fileName) {
        try {
            return ((MatrixWrapper) SerializationUtils.deserialize(fileName)).matrix;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
}
