package preprocessing;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class PngLoader {

    private static final int IMAGE_SIZE = 28;
    private static final int DIGIT_SIZE = 20;

    public static double[] load(String path) throws IOException {

        BufferedImage image = ImageIO.read(new File(path));

        if (image == null) {
            throw new IOException("Immagine non valida: " + path);
        }

        if (image.getWidth() != 28 || image.getHeight() != 28) {
            throw new IllegalArgumentException(
                    "L'immagine deve essere 28x28"
            );
        }

        /*
         * 1. Trova il bounding box della cifra bianca.
         */
        Rectangle bounds = findBounds(image);

        if (bounds == null) {
            throw new IOException(
                    "Nessun pixel della cifra trovato"
            );
        }

        /*
         * 2. Ritaglia solamente la cifra.
         */
        BufferedImage cropped = image.getSubimage(
                bounds.x,
                bounds.y,
                bounds.width,
                bounds.height
        );

        /*
         * 3. Ridimensiona mantenendo le proporzioni.
         *
         * Il lato maggiore diventa 20 pixel.
         */
        double scale = Math.min(
                (double) DIGIT_SIZE / cropped.getWidth(),
                (double) DIGIT_SIZE / cropped.getHeight()
        );

        int newWidth = Math.max(
                1,
                (int) Math.round(cropped.getWidth() * scale)
        );

        int newHeight = Math.max(
                1,
                (int) Math.round(cropped.getHeight() * scale)
        );

        /*
         * 4. Nuova immagine 28x28 nera.
         */
        BufferedImage normalized =
                new BufferedImage(
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                        BufferedImage.TYPE_BYTE_GRAY
                );

        Graphics2D g = normalized.createGraphics();

        g.setColor(Color.BLACK);
        g.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);

        g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );

        /*
         * Prima centratura geometrica.
         */
        int x = (IMAGE_SIZE - newWidth) / 2;
        int y = (IMAGE_SIZE - newHeight) / 2;

        g.drawImage(
                cropped,
                x,
                y,
                newWidth,
                newHeight,
                null
        );

        g.dispose();

        /*
         * 5. Conversione in double[784].
         */
        double[] vector = toVector(normalized);

        /*
         * 6. Centratura usando il centro di massa.
         */
        return centerByMass(vector);
    }


    private static Rectangle findBounds(BufferedImage image) {

        int minX = IMAGE_SIZE;
        int minY = IMAGE_SIZE;

        int maxX = -1;
        int maxY = -1;

        for (int y = 0; y < IMAGE_SIZE; y++) {

            for (int x = 0; x < IMAGE_SIZE; x++) {

                double brightness =
                        getBrightness(image, x, y);

                /*
                 * Sfondo nero = 0
                 * cifra bianca = valori elevati.
                 *
                 * 20 evita di considerare piccoli rumori.
                 */
                if (brightness > 20) {

                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);

                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        if (maxX == -1) {
            return null;
        }

        return new Rectangle(
                minX,
                minY,
                maxX - minX + 1,
                maxY - minY + 1
        );
    }


    private static double[] toVector(
            BufferedImage image
    ) {

        double[] vector =
                new double[IMAGE_SIZE * IMAGE_SIZE];

        for (int y = 0; y < IMAGE_SIZE; y++) {

            for (int x = 0; x < IMAGE_SIZE; x++) {

                double brightness =
                        getBrightness(image, x, y);

                /*
                 * Nero   -> 0
                 * Bianco -> 1
                 */
                vector[y * IMAGE_SIZE + x] =
                        brightness / 255.0;
            }
        }

        return vector;
    }


    private static double getBrightness(
            BufferedImage image,
            int x,
            int y
    ) {

        Color color =
                new Color(image.getRGB(x, y));

        return
                0.299 * color.getRed()
                + 0.587 * color.getGreen()
                + 0.114 * color.getBlue();
    }


    private static double[] centerByMass(
            double[] input
    ) {

        double totalMass = 0.0;

        double weightedX = 0.0;
        double weightedY = 0.0;


        for (int y = 0; y < IMAGE_SIZE; y++) {

            for (int x = 0; x < IMAGE_SIZE; x++) {

                double value =
                        input[y * IMAGE_SIZE + x];

                totalMass += value;

                weightedX += x * value;
                weightedY += y * value;
            }
        }


        if (totalMass == 0.0) {
            return input;
        }


        double centerX =
                weightedX / totalMass;

        double centerY =
                weightedY / totalMass;


        /*
         * Centro di una matrice 28x28:
         * coordinate 0...27
         *
         * centro = 13.5
         */
        double target = 13.5;


        int shiftX =
                (int) Math.round(
                        target - centerX
                );

        int shiftY =
                (int) Math.round(
                        target - centerY
                );


        double[] centered =
                new double[IMAGE_SIZE * IMAGE_SIZE];


        for (int y = 0; y < IMAGE_SIZE; y++) {

            for (int x = 0; x < IMAGE_SIZE; x++) {

                int newX = x + shiftX;
                int newY = y + shiftY;


                if (
                        newX >= 0 &&
                        newX < IMAGE_SIZE &&
                        newY >= 0 &&
                        newY < IMAGE_SIZE
                ) {

                    centered[
                            newY * IMAGE_SIZE + newX
                    ] =
                            input[
                                    y * IMAGE_SIZE + x
                            ];
                }
            }
        }

        return centered;
    }
}