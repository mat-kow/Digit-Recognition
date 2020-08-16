package recognition;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class BackPropagation {
    private static final double ETA = 0.8;
    private static final int MAX_GENERATIONS = 500;


    public static void main(String[] args) {
        learn(16, 2);
    }


    public static void learn(int hiddenLayerSize, int hiddenLayersCount) {
        File dir = new File("D:\\java\\data");
        if (dir.isDirectory()) {
            File[] files = dir.listFiles();
            List<DigitImage> images = Arrays.stream(files).map(DigitImage::new).collect(Collectors.toList());
            Collections.shuffle(images);

            //learning only random 20% of data
            List<DigitImage> part = images.stream().limit(14000).collect(Collectors.toList());
            learn(part, hiddenLayerSize, hiddenLayersCount);
            Collections.shuffle(images);
        } else {
            System.out.println("No files");
        }
    }

    public static void learn(List<DigitImage> digitImages, int hiddenLayerSize, int hiddenLayersCount) {

        int pixelCount = 28 * 28;
        int outputNeuronsCount = 10;

        List<double[][]> weights = new ArrayList<>();
        List<double[][]> biases = new ArrayList<>();

        //        creating random weights and biases for all layers
        for (int i = 0; i < hiddenLayersCount + 1; i++) {
            int inputSize = i == 0 ? pixelCount : hiddenLayerSize;
            int outSize = i < hiddenLayersCount ? hiddenLayerSize : outputNeuronsCount;
            double[][] weightsLayer = new double[outSize][inputSize];
            randomWeights(weightsLayer);
            weights.add(weightsLayer);
            double[][] biasesLayer = new double[outSize][1];
            randomWeights(biasesLayer);
            biases.add(biasesLayer);
        }
        // use this when you want to use "pre-learned" weights and biases
//        try {
//            weights = (List<double[][]>) SerializationUtils.deserialize("D:\\java\\Digit Recognition\\weightsBackProp");
//            biases = (List<double[][]>) SerializationUtils.deserialize("D:\\java\\Digit Recognition\\biasesBackProp");
//        } catch (IOException | ClassNotFoundException e) {
//            System.out.println("weights files not loaded");
//            e.printStackTrace();
//            return;
//        }

        int count = MAX_GENERATIONS;
        while (count-- > 0) {
            double cost = 0;
            List<double[][]> weightsDeltasList = new ArrayList<>();
            List<double[][]> biasDeltasList = new ArrayList<>();
            for (DigitImage digitImage : digitImages) {
                //calculate neurons
                List<double[][]> outputs = new ArrayList<>();
                outputs.add(digitImage.getImage());
                double[][] in = digitImage.getImage();
                for (int i = 0; i < hiddenLayersCount + 1; i++) {
                    double[][] out = calculateOutput(in, weights.get(i), biases.get(i));
                    outputs.add(out);
                    in = out;
                }
                //cost
                for (int i = 0; i < 10; i++) {
                    double ideal = i == digitImage.getDigit() ? 1 : 0;
                    cost += Math.pow(ideal - in[i][0], 2);
                }//cost

                Collections.reverse(outputs);

                Collections.reverse(weights);
                Collections.reverse(biases);
                double[][] errorsNextLayer = null;
                for (int iL = 0; iL < hiddenLayersCount + 1; iL++) {//iteration over layers
                    //neurons delta
                    double[][] neuronsVals = outputs.get(iL);
                    double[][] errorsLayer = new double[neuronsVals.length][1];
                    for (int i = 0; i < neuronsVals.length; i++) {
                        double neuronVal = neuronsVals[i][0];
                        double error;
                        if (iL == 0) {
                            double ideal = i == digitImage.getDigit() ? 1 : 0;
                            error = (ideal - neuronVal);
                        } else {
                            error = Matrix.multiplyMatrixByMatrix(extractColumnAsRow(weights.get(iL - 1), i), errorsNextLayer)[0][0];
                        }
                        double deltaNeuron = (1 - neuronVal) * neuronVal * error;
                        errorsLayer[i][0] = deltaNeuron;
                    }

                    //weight delta
                    int inSize = iL == hiddenLayersCount ? pixelCount : hiddenLayerSize;
                    int outSize = iL == 0 ? outputNeuronsCount : hiddenLayerSize;
                    double[][] deltasWeights = new double[outSize][inSize];
                    for (int j = 0; j < outSize; j++) {
                        //iL indicates layer on which we calculate delta, we need next layer here
                        neuronsVals = outputs.get(iL + 1);
                        for (int i = 0; i < inSize; i++) {
                            double deltaW = ETA * neuronsVals[i][0] * errorsLayer[j][0];
                            deltasWeights[j][i] = deltaW;
                        }
                    }
                    errorsNextLayer = errorsLayer;
                    //bias delta
                    double[][] deltaBias = Matrix.multiplyMatrixByScalar(errorsLayer, ETA);


                    try {
                        double[][] sumDeltasWeight = weightsDeltasList.get(iL);
                        sumDeltasWeight = Matrix.addMatrix(sumDeltasWeight, deltasWeights);
                        weightsDeltasList.set(iL, sumDeltasWeight);

                        double[][] sumDeltasBias = biasDeltasList.get(iL);
                        sumDeltasBias = Matrix.addMatrix(sumDeltasBias, deltaBias);
                        biasDeltasList.set(iL, sumDeltasBias);
                    } catch (IndexOutOfBoundsException e) {//on first iteration
                        weightsDeltasList.add(deltasWeights);

                        biasDeltasList.add(deltaBias);
                    }

                    double[][] newWeights = Matrix.addMatrix(weights.get(iL), deltasWeights);
                    weights.set(iL, newWeights);

                    double[][] newBias = Matrix.addMatrix(biases.get(iL), deltaBias);
                    biases.set(iL, newBias);
                }//layers
                Collections.reverse(weights);
                Collections.reverse(biases);
            }// for image

            //weights update
//            Collections.reverse(weightsDeltasList);
//            for (int i = 0; i < hiddenLayersCount + 1; i++) {
//                double[][] w = weights.get(i);
//                double[][] d = weightsDeltasList.get(i);
//                d = Matrix.multiplyMatrixByScalar(d, 1.0 / digitImages.size());
//                w = Matrix.addMatrix(w, d);
//                weights.set(i, w);
//            }
//
//            Collections.reverse(biasDeltasList);
//            for (int i = 0; i < hiddenLayersCount + 1; i++) {
//                double[][] b = biases.get(i);
//                double[][] d = biasDeltasList.get(i);
//                d = Matrix.multiplyMatrixByScalar(d, 1.0 / digitImages.size());
//                b = Matrix.addMatrix(b, d);
//                biases.set(i, b);
//            }
            //cost
            double avgCost = cost / digitImages.size();
            System.out.println(MAX_GENERATIONS - count + ": " + avgCost);
            if (avgCost < 0.02) {
                break;
            }
        }//while

        //save to file
        try {
            SerializationUtils.serialize(weights, "weightsBackProp");
            SerializationUtils.serialize(biases, "biasesBackProp");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static double[][] calculateOutput(double[][] inputVector, double[][] weights, double[][] biasVector) {
        double[][] output = Matrix.addMatrix(Matrix.multiplyMatrixByMatrix(weights, inputVector), biasVector);
        sigmoidMatrix(output);
        return output;
    }

    private static double[][] extractColumnAsRow(double[][] matrix, int colNo) {
        double[][] result = new double[1][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[0][i] = matrix[i][colNo];
        }
        return result;
    }
    private static double sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, - x));
    }

    public static void randomWeights(double[][] weights) {
        Random random = new Random(new Date().getTime());
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] = random.nextGaussian();
            }
        }
    }

    public static void sigmoidMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = sigmoid(matrix[i][j]);
            }
        }
    }
    public static double[][] arrayToVector(double[] array) {
        double[][] vector = new double[array.length][1];
        for (int i = 0; i < array.length; i++) {
            vector[i][0] = array[i];
        }
        return vector;
    }

    private static List<DigitImage> data3x5() {
        double[][] digits = new double[10][15];
        digits[0] = new double[]{1, 1, 1,   1, 0, 1,    1, 0, 1,    1, 0, 1,    1, 1, 1};
        digits[1] = new double[]{0, 1, 0,   0, 1, 0,    0, 1, 0,    0, 1, 0,    0, 1, 0};
        digits[2] = new double[]{1, 1, 1,   0, 0, 1,    1, 1, 1,    1, 0, 0,    1, 1, 1};
        digits[3] = new double[]{1, 1, 1,   0, 0, 1,    1, 1, 1,    0, 0, 1,    1, 1, 1};
        digits[4] = new double[]{1, 0, 1,   1, 0, 1,    1, 1, 1,    0, 0, 1,    0, 0, 1};
        digits[5] = new double[]{1, 1, 1,   1, 0, 0,    1, 1, 1,    0, 0, 1,    1, 1, 1};
        digits[6] = new double[]{1, 1, 1,   1, 0, 0,    1, 1, 1,    1, 0, 1,    1, 1, 1};
        digits[7] = new double[]{1, 1, 1,   0, 0, 1,    0, 0, 1,    0, 0, 1,    0, 0, 1};
        digits[8] = new double[]{1, 1, 1,   1, 0, 1,    1, 1, 1,    1, 0, 1,    1, 1, 1};
        digits[9] = new double[]{1, 1, 1,   1, 0, 1,    1, 1, 1,    0, 0, 1,    1, 1, 1};
        List<DigitImage> digitImages = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            digitImages.add(new DigitImage(arrayToVector(digits[i]), i));
        }
        digitImages.add(new DigitImage(arrayToVector(
                new double[]{0, 1, 0,   0, 1, 0,    1, 1, 0,    1, 1, 0,    0, 1, 1}), 1));
        digitImages.add(new DigitImage(arrayToVector(
                new double[]{1, 1, 1,   1, 0, 1,    0, 0, 1,    0, 0, 1,    0, 0, 1}), 7));
        digitImages.add(new DigitImage(arrayToVector(
                new double[]{1, 1, 0,   0, 0, 1,    0, 0, 1,    1, 0, 0,    1, 1, 1}), 2));
        digitImages.add(new DigitImage(arrayToVector(
                new double[]{1, 1, 1,   1, 0, 1,    1, 1, 1,    0, 0, 1,    0, 1, 1}), 9));
        digitImages.add(new DigitImage(arrayToVector(
                new double[]{0, 1, 1,   1, 0, 0,    1, 1, 1,    1, 0, 1,    1, 1, 1}), 6));
        return digitImages;

    }
}
