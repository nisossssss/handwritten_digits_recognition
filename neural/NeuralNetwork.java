package neural;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class NeuralNetwork {

    private Layer hiddenLayer;
    private Layer outputLayer;

    public NeuralNetwork(){

        hiddenLayer = new Layer(128, 784);
        outputLayer = new Layer(10, 128);
    }
    

    public double[] forward(double[] input){
        double[] hiddenOutputs = hiddenLayer.forward(input, true);

        double[] logits = outputLayer.forward(hiddenOutputs, false);

        return Activation.softmax(logits);
    }

    public void save(String path) throws IOException {

    try (DataOutputStream out = new DataOutputStream(
            new BufferedOutputStream(
                    new FileOutputStream(path)
            )
    )) {

        saveLayer(out, hiddenLayer);
        saveLayer(out, outputLayer);
    }
}

private void saveLayer(
            DataOutputStream out,
            Layer layer
    ) throws IOException {

        Neuron[] neurons = layer.getNeurons();

        out.writeInt(neurons.length);
        out.writeInt(neurons[0].getWeights().length);

        for (Neuron neuron : neurons) {

            for (double weight : neuron.getWeights()) {
                out.writeDouble(weight);
            }

            out.writeDouble(neuron.getBias());
        }
    }

    public void load(String path) throws IOException {

    try (DataInputStream in = new DataInputStream(
            new BufferedInputStream(
                    new FileInputStream(path)
            )
    )) {

        loadLayer(in, hiddenLayer);
        loadLayer(in, outputLayer);
    }
}

private void loadLayer(DataInputStream in, Layer layer) throws IOException {

        int numberOfNeurons = in.readInt();
        int numberOfWeights = in.readInt();

        Neuron[] neurons = layer.getNeurons();

        if (neurons.length != numberOfNeurons) {
            throw new IOException("Architettura incompatibile");
        }

        if (neurons[0].getWeights().length != numberOfWeights) {
            throw new IOException("Architettura incompatibile");
        }

        for (Neuron neuron : neurons) {

            for (int i = 0; i < numberOfWeights; i++) {
                neuron.setWeight(i, in.readDouble());
            }

            neuron.setBias(in.readDouble());
        }
    }

    public int predict(double[] input) {
        double[] probabilities = forward(input);

        int prediction = 0;

        for (int i = 1; i < probabilities.length; i = i + 1){
            if(probabilities[i] > probabilities[prediction]) prediction = i;
        }

        return prediction;
    }

    public Layer getHiddenLayer() {
        return hiddenLayer;
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }
}
