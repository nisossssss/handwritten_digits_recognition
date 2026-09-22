package data.mnist;

public class MnistSample{

    private final double[] pixels;
    private final int label;

    public MnistSample(double[] pixels, int label){
        this.pixels = pixels;
        this.label = label;
    }

    public double[] getPixels(){ return pixels; }
    public int getLabel() {return label; }

}