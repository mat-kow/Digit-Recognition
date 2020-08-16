package recognition;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    private static final Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        int hiddenLayerCount = 2;
        while (true) {
            System.out.println("0. Exit\n" +
                    "1. Learn the network\n" +
                    "2. Guess all the numbers\n" +
                    "3. Guess number from text file\n" +
                    "4. Guess number from console input");
            System.out.print("Your choice: ");
            String action = scanner.nextLine();
            switch (action) {
                case "1":
                    System.out.print("How many hidden layers: ");
                    hiddenLayerCount = Integer.parseInt(scanner.nextLine());
                    System.out.print("Size of hidden layers: ");
                    int hiddenLayerSize = Integer.parseInt(scanner.nextLine());
                    System.out.println("Learning...");
                    BackPropagation.learn(hiddenLayerSize, hiddenLayerCount);
                    System.out.println("Done! Saved to the file.");
                    break;
                case "2":
                    System.out.println("Guessing...");
                    guessAll(hiddenLayerCount);
                    break;
                case "0":
                    return;
                case "4":
                    System.out.println("Enter 28 lines, with 28 numbers each");
                    double[][] image = new double[28 * 28][1];
                    for (int i = 0; i < 28; i++) {
                        String[] line = scanner.nextLine().trim().split("\\s+");
                        List<Integer> ints = Arrays.stream(line).map(s -> Integer.parseInt(s)).collect(Collectors.toList());
                        for (int j = 0; j < 28; j++) {
                            image[i * 28 + j][0] = ints.get(j) / 255.0;
                        }
                    }

                    DigitImage digitImagef = new DigitImage(image, -1);
                    guessBackProp(digitImagef);

                    return;
                case "3":
                    System.out.print("Enter filename: ");
                    String filename = scanner.nextLine();
                    File file = new File(filename);
                    DigitImage digitImage = new DigitImage(file);
                    guessBackProp(digitImage);
                    return;
            }//switch
        }//while

    }

    private static void guessAll(int hiddenLayersCount) {
        File dir = new File("D:\\java\\data");
        List<DigitImage> images;
        if (dir.isDirectory()) {
            File[] files = dir.listFiles();
            images = Arrays.stream(files).map(DigitImage::new).collect(Collectors.toList());
        } else {
            System.out.println("No files");
            return;
        }
        int right = 0;
        List<double[][]> weights;
        List<double[][]> biases;
        try {
            weights = (List<double[][]>) SerializationUtils.deserialize("weightsBackProp");
            biases = (List<double[][]>) SerializationUtils.deserialize("biasesBackProp");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }

        for (DigitImage image : images) {
            double[][] input = image.getImage();
            for (int i = 0; i < hiddenLayersCount + 1; i++) {
                input = BackPropagation.calculateOutput(input, weights.get(i), biases.get(i));
            }
            double max = input[0][0];
            for (int i = 1; i < 10; i++) {
                max = Math.max(max, input[i][0]);
            }
            int index = -1;
            for (int i = 0; i < 10; i++) {
                if (input[i][0] == max) {
                    index = i;
                    break;
                }
            }
            if (image.getDigit() == index) {
                right++;
            }

        }
        System.out.printf("The network prediction accuracy: %d/70000, %d", right, right / 700);
        System.out.println("%");

    }


    private static void guessBackProp(DigitImage digitImage) {
        List<double[][]> weights;
        List<double[][]> biases;
        try {
            weights = (List<double[][]>) SerializationUtils.deserialize("weightsBackProp");
            biases = (List<double[][]>) SerializationUtils.deserialize("biasesBackProp");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }
        int hiddenLayersCount = 2;
        double[][] input = digitImage.getImage();
        for (int i = 0; i < hiddenLayersCount + 1; i++) {
            input = BackPropagation.calculateOutput(input, weights.get(i), biases.get(i));
        }
        double max = input[0][0];
        for (int i = 1; i < 10; i++) {
            max = Math.max(max, input[i][0]);
        }
        int index = -1;
        double[] forPrint = new double[10];
        for (int i = 0; i < 10; i++) {
            if (input[i][0] == max) {
                index = i;
            }
            forPrint[i] = Math.round(input[i][0] * 1000) / 1000.0;
        }
        System.out.println(Arrays.toString(forPrint));
        System.out.println("This number is " + index);

    }

}
