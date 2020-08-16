package recognition;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

public class DigitImage implements Serializable {
    private static final long serialVersionUID = 1L;

    private double[][] image;
    private int digit;

    public DigitImage(double[][] image, int digit) {
        this.image = image;
        this.digit = digit;
    }

    public DigitImage(File file) {
        image = new double[28 * 28][1];
        try (Scanner scanner = new Scanner(file)){
            for (int i = 0; i < 28; i++) {
                String[] line = scanner.nextLine().trim().split("\\s+");
                List<Integer> ints = Arrays.stream(line).map(s -> Integer.parseInt(s)).collect(Collectors.toList());
                for (int j = 0; j < 28; j++) {
                    image[i * 28 + j][0] = ints.get(j) / 255.0;
                }
            }
            digit = Integer.parseInt(scanner.nextLine());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public double[][] getImage() {
        return image;
    }

    public int getDigit() {
        return digit;
    }
}
