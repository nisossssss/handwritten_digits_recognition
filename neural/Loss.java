package neural;

public class Loss {

    public static double crossEntropy(double[] probabilities, int correctLabel){
        double probability = probabilities[correctLabel];

        probability = Math.max(probability, 1e-15);
        return -Math.log(probability);
    }
    
}
