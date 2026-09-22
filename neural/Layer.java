package neural;

public class Layer {
    private Neuron[] neurons;

    public Layer(int neurons, int inputs){
        this.neurons = new Neuron[neurons];

        for(int i = 0; i < neurons; i = i + 1) this.neurons[i] = new Neuron(inputs);
    }

    public double[] forward(double[] inputs, boolean useRelu) {
        double[] outputs = new double[neurons.length];

        for(int i = 0; i < neurons.length; i = i + 1){
            double z = neurons[i].calculate(inputs);

            double output;
            if(useRelu) output = Activation.relu(z);
            else output = z;

            neurons[i].setOutput(output);
            outputs[i] = output;
        }

        return outputs;
    }

    public Neuron[] getNeurons() { return neurons; }
}
