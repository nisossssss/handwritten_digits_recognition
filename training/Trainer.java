package training;

import neural.*;

public class Trainer {
    public static double trainSample(NeuralNetwork network, double[] input, int label, double learningRate){

        Layer hiddenLayer = network.getHiddenLayer();
        Layer outputLayer = network.getOutputLayer();

        // forward propagation

        double[] hiddenOutputs = hiddenLayer.forward(input, true);
        double[] logits = outputLayer.forward(hiddenOutputs, false);
        double[] probabilites = Activation.softmax(logits);

        // LOSS

        double probabilityCorrect = probabilites[label];
        double loss = -Math.log(Math.max(probabilityCorrect, 1e-15));

        // output deltas (delta_k = p_k - y_k)

        Neuron[] outputNeurons = outputLayer.getNeurons();
        double[] outputDeltas = new double[outputNeurons.length];

        for(int k = 0; k < outputNeurons.length; k = k + 1){
            double target = (k == label) ? 1.0 : 0.0;
            outputDeltas[k] = probabilites[k] - target;
        }

        // hidden deltas
        Neuron[] hiddenNeurons = hiddenLayer.getNeurons();
        double[] hiddenDeltas = new double[hiddenNeurons.length];

        for(int j = 0; j < hiddenNeurons.length; j = j + 1){
            double error = 0.0;
            for(int k = 0; k < outputNeurons.length; k = k + 1){
                double weight = outputNeurons[k].getWeights()[j];
                error += weight * outputDeltas[k];
            }

            hiddenDeltas[j] = Activation.reluDerivative(hiddenNeurons[j].getZ()) * error;
        }

        // update output layer

        for(int k = 0; k < outputNeurons.length; k = k + 1){
            Neuron neuron = outputNeurons[k];
            for(int j = 0; j < hiddenOutputs.length; j = j + 1){
                double gradient = outputDeltas[k] * hiddenOutputs[j];

                neuron.updateWeight(j, -learningRate * gradient);
            }
        }

        // update hidden layer

        for(int j = 0; j < hiddenNeurons.length; j = j + 1){
            Neuron neuron = hiddenNeurons[j];
            for(int i = 0; i < input.length; i = i + 1){
                double gradient = hiddenDeltas[j] * input[i];
                neuron.updateWeight(i, -learningRate * gradient);
                neuron.updateBias(-learningRate * hiddenDeltas[j]);
            }
        }
        return loss;
    }
}
