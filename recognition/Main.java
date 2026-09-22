import data.mnist.*;

import neural.NeuralNetwork;

import preprocessing.Augmentation;
import preprocessing.PngLoader;

import training.Trainer;

import java.io.File;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


public class Main {

    /*
     * ============================================================
     * PATH
     * ============================================================
     */

    private static final String MODEL_PATH =
            "models/mnist-model.bin";


    private static final String TRAIN_IMAGES =
            "data/mnist/train-images.idx3-ubyte";

    private static final String TRAIN_LABELS =
            "data/mnist/train-labels.idx1-ubyte";


    private static final String TEST_IMAGES =
            "data/mnist/t10k-images.idx3-ubyte";

    private static final String TEST_LABELS =
            "data/mnist/t10k-labels.idx1-ubyte";


    private static final String MY_DIGITS_FOLDER =
            "data/my_digits";


    /*
     * ============================================================
     * TRAINING PARAMETERS
     * ============================================================
     */

    private static final double LEARNING_RATE = 0.003;

    private static final int EPOCHS = 10;

    /*
     * Probabilità di applicare augmentation.
     *
     * 0.5 = 50%
     */
    private static final double AUGMENTATION_PROBABILITY = 0.5;


    /*
     * ============================================================
     * MAIN
     * ============================================================
     */

    public static void main(String[] args) throws Exception {

        if (args.length == 0) {

            printUsage();

            return;
        }


        switch (args[0]) {

            /*
             * TRAINING
             */
            case "0":
                train();
                break;


            /*
             * RECOGNITION
             */
            case "1":
                recognize();
                break;


            default:

                System.out.println(
                        "Modalità non valida: " + args[0]
                );

                System.out.println();

                printUsage();
        }
    }


    /*
     * ============================================================
     * TRAINING
     * ============================================================
     */

    private static void train() throws Exception {

        System.out.println(
                "=== TRAINING MODE ==="
        );

        System.out.println();


        /*
         * --------------------------------------------------------
         * CARICAMENTO MNIST
         * --------------------------------------------------------
         */

        List<MnistSample> trainingSet =
                MnistLoader.load(
                        TRAIN_IMAGES,
                        TRAIN_LABELS
                );


        List<MnistSample> testSet =
                MnistLoader.load(
                        TEST_IMAGES,
                        TEST_LABELS
                );


        /*
         * --------------------------------------------------------
         * CREAZIONE RETE
         *
         * 784 -> 128 -> 10
         * --------------------------------------------------------
         */

        NeuralNetwork network =
                new NeuralNetwork();


        /*
         * Migliore accuracy trovata.
         */
        double bestAccuracy = 0.0;

        int bestEpoch = 0;


        /*
         * --------------------------------------------------------
         * CARTELLA MODEL
         * --------------------------------------------------------
         */

        File modelDirectory =
                new File("models");


        if (!modelDirectory.exists()) {

            if (!modelDirectory.mkdirs()) {

                throw new RuntimeException(
                        "Impossibile creare la cartella models"
                );
            }
        }


        /*
         * ========================================================
         * EPOCH
         * ========================================================
         */

        for (
                int epoch = 1;
                epoch <= EPOCHS;
                epoch++
        ) {

            /*
             * Ogni epoch cambiamo l'ordine
             * delle immagini.
             */
            Collections.shuffle(
                    trainingSet
            );


            double totalLoss = 0.0;


            System.out.println();

            System.out.println(
                    "Inizio epoch " + epoch
            );


            /*
             * ====================================================
             * TRAINING SET
             * ====================================================
             */

            for (
                    int i = 0;
                    i < trainingSet.size();
                    i++
            ) {

                MnistSample sample =
                        trainingSet.get(i);


                /*
                 * Immagine originale.
                 */
                double[] input =
                        sample.getPixels();


                /*
                 * ------------------------------------------------
                 * DATA AUGMENTATION
                 * ------------------------------------------------
                 *
                 * 50%:
                 *
                 * immagine originale
                 *
                 * 50%:
                 *
                 * immagine spostata casualmente
                 * tra -2 e +2 pixel
                 *
                 */

                if (
        Math.random()
                < AUGMENTATION_PROBABILITY
            ) {

                input =
                        Augmentation.randomTransform(
                                input
                        );
            }


                /*
                 * ------------------------------------------------
                 * TRAINING DI UN SAMPLE
                 *
                 * forward
                 * loss
                 * backpropagation
                 * gradient descent
                 * ------------------------------------------------
                 */

                double loss =
                        Trainer.trainSample(
                                network,
                                input,
                                sample.getLabel(),
                                LEARNING_RATE
                        );


                totalLoss += loss;


                /*
                 * Progress output.
                 */
                if ((i + 1) % 10000 == 0) {

                    System.out.println(
                            "  Processate "
                                    + (i + 1)
                                    + " / "
                                    + trainingSet.size()
                                    + " immagini"
                    );
                }
            }


            /*
             * ====================================================
             * STATISTICHE EPOCH
             * ====================================================
             */

            double averageLoss =
                    totalLoss
                            / trainingSet.size();


            /*
             * ATTENZIONE:
             *
             * evaluate() NON modifica i pesi.
             *
             * Fa solamente:
             *
             * forward -> prediction -> confronto label
             */

            double trainAccuracy =
                    evaluate(
                            network,
                            trainingSet
                    );


            double testAccuracy =
                    evaluate(
                            network,
                            testSet
                    );


            System.out.printf(
                    "Epoch %d completata | " +
                            "Loss: %.4f | " +
                            "Train: %.2f%% | " +
                            "Test: %.2f%%%n",

                    epoch,

                    averageLoss,

                    trainAccuracy * 100,

                    testAccuracy * 100
            );


            /*
             * ====================================================
             * SALVATAGGIO MIGLIOR MODELLO
             * ====================================================
             *
             * Salviamo SOLO se questa rete
             * è migliore di tutte le precedenti.
             */

            if (
                    testAccuracy
                            > bestAccuracy
            ) {

                bestAccuracy =
                        testAccuracy;

                bestEpoch =
                        epoch;


                network.save(
                        MODEL_PATH
                );


                System.out.printf(
                        ">>> Nuovo miglior modello salvato! " +
                                "Accuracy: %.2f%%%n",

                        bestAccuracy * 100
                );
            }
        }


        /*
         * ========================================================
         * FINE TRAINING
         * ========================================================
         */

        System.out.println();

        System.out.println(
                "=============================="
        );

        System.out.println(
                "Training completato."
        );


        System.out.println(
                "Migliore epoch: "
                        + bestEpoch
        );


        System.out.printf(
                "Migliore accuracy: %.2f%%%n",
                bestAccuracy * 100
        );


        System.out.println(
                "Miglior modello salvato in:"
        );

        System.out.println(
                MODEL_PATH
        );


        System.out.println(
                "=============================="
        );
    }


    /*
     * ============================================================
     * RECOGNITION MODE
     * ============================================================
     */

    private static void recognize() throws Exception {

        System.out.println(
                "=== RECOGNITION MODE ==="
        );

        System.out.println();


        /*
         * --------------------------------------------------------
         * CONTROLLO MODELLO
         * --------------------------------------------------------
         */

        File modelFile =
                new File(
                        MODEL_PATH
                );


        if (!modelFile.exists()) {

            System.out.println(
                    "Modello non trovato:"
            );

            System.out.println(
                    MODEL_PATH
            );


            System.out.println();

            System.out.println(
                    "Esegui prima:"
            );

            System.out.println(
                    "java Main 0"
            );

            return;
        }


        /*
         * --------------------------------------------------------
         * CARICAMENTO RETE
         * --------------------------------------------------------
         */

        NeuralNetwork network =
                new NeuralNetwork();


        network.load(
                MODEL_PATH
        );


        System.out.println(
                "Modello caricato: "
                        + MODEL_PATH
        );


        /*
         * --------------------------------------------------------
         * CARTELLA PNG
         * --------------------------------------------------------
         */

        File folder =
                new File(
                        MY_DIGITS_FOLDER
                );


        if (
                !folder.exists()
                        ||
                !folder.isDirectory()
        ) {

            System.out.println(
                    "Cartella non trovata:"
            );

            System.out.println(
                    MY_DIGITS_FOLDER
            );

            return;
        }


        /*
         * --------------------------------------------------------
         * CERCA TUTTI I PNG
         * --------------------------------------------------------
         */

        File[] images =
                folder.listFiles(
                        (dir, name) ->
                                name
                                        .toLowerCase()
                                        .endsWith(".png")
                );


        if (
                images == null
                        ||
                images.length == 0
        ) {

            System.out.println(
                    "Nessuna immagine PNG trovata in:"
            );

            System.out.println(
                    MY_DIGITS_FOLDER
            );

            return;
        }


        /*
         * Ordine alfabetico.
         *
         * 0.png
         * 1.png
         * 2.png
         * ...
         */
        Arrays.sort(
                images,
                Comparator.comparing(
                        File::getName
                )
        );


        System.out.println();

        System.out.println(
                "Trovate "
                        + images.length
                        + " immagini."
        );


        /*
         * ========================================================
         * RICONOSCIMENTO
         * ========================================================
         */

        for (File image : images) {

            recognizeImage(
                    network,
                    image
            );
        }
    }


    /*
     * ============================================================
     * RICONOSCIMENTO SINGOLA IMMAGINE
     * ============================================================
     */

    private static void recognizeImage(
            NeuralNetwork network,
            File image
    ) throws Exception {


        /*
         * --------------------------------------------------------
         * PNG -> double[784]
         *
         * PngLoader si occupa del preprocessing.
         * --------------------------------------------------------
         */

        double[] input =
                PngLoader.load(
                        image.getPath()
                );


        /*
         * --------------------------------------------------------
         * FORWARD PROPAGATION
         * --------------------------------------------------------
         */

        double[] probabilities =
                network.forward(
                        input
                );


        /*
         * --------------------------------------------------------
         * CLASSE CON PROBABILITÀ MASSIMA
         * --------------------------------------------------------
         */

        int prediction =
                argMax(
                        probabilities
                );


        /*
         * --------------------------------------------------------
         * OUTPUT
         * --------------------------------------------------------
         */

        System.out.println();

        System.out.println();

        System.out.println(
                "================================"
        );


        System.out.println(
                "Immagine: "
                        + image.getName()
        );


        System.out.println(
                "================================"
        );


        /*
         * Visualizziamo esattamente ciò
         * che entra nella rete.
         */
        printImage(
                input
        );


        System.out.println();

        System.out.println(
                "Probabilità:"
        );

        System.out.println();


        for (
                int i = 0;
                i < probabilities.length;
                i++
        ) {

            System.out.printf(
                    "%d -> %6.2f%%%n",

                    i,

                    probabilities[i]
                            * 100
            );
        }


        System.out.println();


        System.out.println(
                "Numero riconosciuto: "
                        + prediction
        );


        System.out.printf(
                "Confidenza: %.2f%%%n",

                probabilities[prediction]
                        * 100
        );
    }


    /*
     * ============================================================
     * EVALUATION
     * ============================================================
     */

    private static double evaluate(
            NeuralNetwork network,
            List<MnistSample> dataset
    ) {

        int correct = 0;


        for (
                MnistSample sample
                        : dataset
        ) {

            /*
             * Solo forward.
             *
             * NON viene effettuato training.
             */
            int prediction =
                    network.predict(
                            sample.getPixels()
                    );


            if (
                    prediction
                            == sample.getLabel()
            ) {

                correct++;
            }
        }


        return (double) correct
                / dataset.size();
    }


    /*
     * ============================================================
     * ARGMAX
     * ============================================================
     */

    private static int argMax(
            double[] values
    ) {

        int maxIndex = 0;


        for (
                int i = 1;
                i < values.length;
                i++
        ) {

            if (
                    values[i]
                            > values[maxIndex]
            ) {

                maxIndex = i;
            }
        }


        return maxIndex;
    }


    /*
     * ============================================================
     * STAMPA IMMAGINE 28x28
     * ============================================================
     */

    private static void printImage(
            double[] pixels
    ) {

        for (
                int row = 0;
                row < 28;
                row++
        ) {

            for (
                    int col = 0;
                    col < 28;
                    col++
            ) {

                double value =
                        pixels[
                                row * 28
                                        + col
                        ];


                if (value > 0.75) {

                    System.out.print(
                            "██"
                    );

                } else if (
                        value > 0.40
                ) {

                    System.out.print(
                            "▓▓"
                    );

                } else if (
                        value > 0.15
                ) {

                    System.out.print(
                            "░░"
                    );

                } else {

                    System.out.print(
                            "  "
                    );
                }
            }


            System.out.println();
        }
    }


    /*
     * ============================================================
     * HELP
     * ============================================================
     */

    private static void printUsage() {

        System.out.println(
                "Utilizzo:"
        );

        System.out.println();

        System.out.println(
                "java Main 0"
        );

        System.out.println(
                "    Training MNIST + salvataggio miglior modello"
        );


        System.out.println();

        System.out.println(
                "java Main 1"
        );

        System.out.println(
                "    Riconoscimento immagini in data/my_digits"
        );
    }
}