package preprocessing;

public class ImageVectorizer {
    public static double[] flatten(double[][] matrix){

        double[] vector = new double[28 * 28];
    
        int index = 0;

        for(int row = 0; row < 28; row = row + 1){
            for(int col = 0; col < 28; col = col + 1){

                vector[index] = matrix[row][col];
                index = index + 1;
            }
        }
        return vector;
    }
}
