package preprocessing;

import java.util.Random;

public class Augmentation {

    private static final int SIZE = 28;
    private static final double CENTER = 13.5;

    private static final Random random = new Random();


    /*
     * Augmentation completa per ora:
     *
     * - scala casuale 90% -> 110%
     * - shift X: -2 -> +2
     * - shift Y: -2 -> +2
     */
    public static double[] randomTransform(double[] input) {

        double scale =
                0.90 + random.nextDouble() * 0.20;

        int shiftX =
                random.nextInt(5) - 2;

        int shiftY =
                random.nextInt(5) - 2;

        return transform(
                input,
                scale,
                shiftX,
                shiftY
        );
    }


    /*
     * Applica contemporaneamente:
     *
     * SCALE
     * SHIFT
     */
    public static double[] transform(
            double[] input,
            double scale,
            int shiftX,
            int shiftY
    ) {

        if (input.length != SIZE * SIZE) {

            throw new IllegalArgumentException(
                    "L'immagine deve contenere 784 pixel"
            );
        }

        double[] output =
                new double[SIZE * SIZE];


        /*
         * Usiamo inverse mapping:
         *
         * per ogni pixel della nuova immagine
         * cerchiamo da quale posizione
         * dell'immagine originale proviene.
         *
         * Questo evita "buchi" nell'immagine.
         */

        for (int y = 0; y < SIZE; y++) {

            for (int x = 0; x < SIZE; x++) {


                /*
                 * Rimuoviamo prima lo shift.
                 */
                double shiftedX =
                        x - shiftX;

                double shiftedY =
                        y - shiftY;


                /*
                 * Riportiamo il punto rispetto
                 * al centro dell'immagine.
                 *
                 * Poi applichiamo l'inverso
                 * della scala.
                 */
                double sourceX =
                        (shiftedX - CENTER)
                                / scale
                                + CENTER;

                double sourceY =
                        (shiftedY - CENTER)
                                / scale
                                + CENTER;


                /*
                 * Interpolazione bilineare.
                 */
                double value =
                        bilinearSample(
                                input,
                                sourceX,
                                sourceY
                        );


                output[y * SIZE + x] =
                        value;
            }
        }


        return output;
    }


    /*
     * ============================================================
     * BILINEAR INTERPOLATION
     * ============================================================
     *
     * sourceX/sourceY possono essere ad esempio:
     *
     * 10.4, 15.7
     *
     * quindi prendiamo i quattro pixel vicini
     * e calcoliamo un valore intermedio.
     */
    private static double bilinearSample(
            double[] input,
            double x,
            double y
    ) {

        /*
         * Fuori immagine = sfondo nero.
         */
        if (
                x < 0 ||
                x > SIZE - 1 ||
                y < 0 ||
                y > SIZE - 1
        ) {

            return 0.0;
        }


        int x0 =
                (int) Math.floor(x);

        int y0 =
                (int) Math.floor(y);


        int x1 =
                Math.min(
                        x0 + 1,
                        SIZE - 1
                );

        int y1 =
                Math.min(
                        y0 + 1,
                        SIZE - 1
                );


        /*
         * Parte decimale.
         */
        double dx =
                x - x0;

        double dy =
                y - y0;


        /*
         * Quattro pixel attorno
         * alla posizione richiesta.
         */

        double p00 =
                input[y0 * SIZE + x0];

        double p10 =
                input[y0 * SIZE + x1];

        double p01 =
                input[y1 * SIZE + x0];

        double p11 =
                input[y1 * SIZE + x1];


        /*
         * Interpolazione orizzontale.
         */

        double top =
                p00 * (1.0 - dx)
                        + p10 * dx;

        double bottom =
                p01 * (1.0 - dx)
                        + p11 * dx;


        /*
         * Interpolazione verticale.
         */

        return
                top * (1.0 - dy)
                        + bottom * dy;
    }


    /*
     * Manteniamo anche lo shift singolo,
     * utile eventualmente per fare test.
     */
    public static double[] shift(
            double[] input,
            int shiftX,
            int shiftY
    ) {

        return transform(
                input,
                1.0,
                shiftX,
                shiftY
        );
    }


    /*
     * Solo scale, utile per debug.
     */
    public static double[] scale(
            double[] input,
            double scale
    ) {

        return transform(
                input,
                scale,
                0,
                0
        );
    }
}