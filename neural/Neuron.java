package neural;

import java.util.Random;

public class Neuron {
    private double[] weights;
    private double bias;

    private double z;
    private double output;

    private static final Random random = new Random();

    public Neuron(int inputs){
        weights = new double[inputs];

        double scale = Math.sqrt(2.0 / inputs);

        for(int i = 0; i < weights.length; i = i + 1) weights[i] = random.nextGaussian() * scale;
        bias = 0.0;
    }

    public void updateWeight(int index, double amount) { weights[index] += amount; }
    public void updateBias(double amount) { bias += amount; }

    

    public double calculate(double[] inputs){
        if(inputs.length != weights.length) throw new IllegalArgumentException("Recived inputs: " + inputs.length + ", expected inputs: " + weights.length);

        z = bias;
        for(int i = 0; i < weights.length; i = i + 1) z+= weights[i] * inputs[i];

        return z;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getOutput() {
        return output;
    }

    public double getZ() {
        return z;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
    

    public void setWeight(int index, double value) {
        weights[index] = value;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
