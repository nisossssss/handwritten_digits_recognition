package neural;

public class Activation {

    public static double relu(double x){
        return Math.max(0.0, x);
    }

    public static double reluDerivative(double z) { return z > 0.0 ? 1.0 : 0.0; }

    public static double[] softmax(double[] logits){
        double max = logits[0];

        for(double value : logits) if(value > max) max = value;

        double[] result = new double[logits.length];
        double sum = 0.0;

        for(int i = 0; i < logits.length; i = i + 1) {
            result[i] = Math.exp(logits[i] - max);
            sum += result[i];
        }
        
        for(int i = 0; i < result.length; i = i + 1) result[i] /= sum;

        return result;
    }
}
