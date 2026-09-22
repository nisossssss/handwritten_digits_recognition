package data.mnist;

import java.util.*;
import java.io.*;


public class MnistLoader {
    private static final int IMAGE_MAGIC_NUMBER = 2051;
    private static final int LABEL_MAGIC_NUMBER = 2049;

    public static List<MnistSample> load ( String imagesPath, String labelsPath) throws IOException{
        List<MnistSample> dataset = new ArrayList<>();

        try
        (
            DataInputStream images = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesPath)));
            DataInputStream labels = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsPath)));
        )
        {
            int imageMagic = images.readInt();
            int numberOfImages = images.readInt();
            int rows = images.readInt();
            int columns = images.readInt();
            
            int labelMagic = labels.readInt();
            int numberOfLabels = labels.readInt();

            if(imageMagic != IMAGE_MAGIC_NUMBER) { throw new IOException("Invalid value image magic number"); }
            if(labelMagic != LABEL_MAGIC_NUMBER) { throw new IOException("Invalid value label magic number"); }
            if(numberOfImages != numberOfLabels) { throw new IOException("Differents number of images and labels"); }
            
            System.out.println("Images: " + numberOfImages);
            System.out.println("Size: " + rows + "x" + columns);

            int imageSize = rows * columns;
        
            for (int n = 0; n < numberOfImages; n++) {

                double[] pixels = new double[imageSize];

                for (int i = 0; i < imageSize; i++) {

                    int pixel = images.readUnsignedByte();

                    pixels[i] = pixel / 255.0;
                }

                int label = labels.readUnsignedByte();

                dataset.add(
                        new MnistSample(pixels, label)
                );
            }
        }

        return dataset;
    }
}
