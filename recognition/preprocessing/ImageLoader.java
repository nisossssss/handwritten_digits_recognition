package preprocessing;

import java.io.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;;

public class ImageLoader {
    public static double[][] load(String path) throws IOException{
        
        BufferedImage image = ImageIO.read(new File(path));

        if(image.getWidth() != 28 || image.getHeight() != 28) throw new IllegalArgumentException(" Image must be 28x28.");

        double[][] pixels = new double[28][28];

        for(int y = 0; y < 28; y = y + 1){
            for(int x = 0; x < 28; x = x + 1){
                int rgb = image.getRGB(x, y);

                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                double gray = (red + green + blue) / 3.0;

                double normalized = gray/255.0;
                
                pixels[y][x] = 1.0 - normalized;
            }
        }
        return pixels;
    }
}
