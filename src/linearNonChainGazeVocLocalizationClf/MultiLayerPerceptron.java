package linearNonChainGazeVocLocalizationClf;

import java.io.Serializable;
import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;


/**
 * Implements backpropagation as described in:
 * http://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
 * @author Andreas Thiele
 */
public class MultiLayerPerceptron implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = -1917800061580464138L;
	/**
	 * Example of the network solving the XOR-problem
	 */
//	public static void main(String[] args){
////		MultiLayerPerceptron mlp = new MultiLayerPerceptron(2, 5, 1, 1, 1.0);
//		MultiLayerPerceptron mlp = new MultiLayerPerceptron(2, 5, 1, 1, 1.0);
//		double[][] xor = new double[4][2];
//		xor[0][0] = 0;
//		xor[0][1] = 0;
//		xor[1][0] = 1;
//		xor[1][1] = 0;
//		xor[2][0] = 0;
//		xor[2][1] = 1;
//		xor[3][0] = 1;
//		xor[3][1] = 1;
//		Random r = new Random();
//		for(int i = 0; i <= 100000; i++){
//			double[] input = xor[r.nextInt(4)];
//			double[] target = new double[]{((int)input[0]+(int)input[1])%2};
//			mlp.train(input, target);
//		}
//		for(int i = 0; i < 4; i++){
//			System.out.println("Classifying "+xor[i][0]+","+xor[i][1]+". Output: "+mlp.classify(xor[i])[0]);
//		}
//	}
	
	private double learningrate;
	private ArrayList<ArrayList<Neuron>> hidden;
	private ArrayList<Neuron> input;
	private ArrayList<Neuron> output;
	private HashMap<Neuron,Integer> inputIndex, outputIndex;
	private HashMap<Integer,HashMap<Neuron,Integer>> hiddenIndex;
	
	/**
	 * 
	 * @param input Size of input layer
	 * @param hidden Size if hidden layer(s)
	 * @param output Size of output layer
	 * @param numberOfHiddenLayers Number of hidden layers
	 * @param learningrate Learning rate
	 */
	public MultiLayerPerceptron(int input, int hidden, int output, int numberOfHiddenLayers, double learningrate){
		this.hiddenIndex = new HashMap<Integer,HashMap<Neuron,Integer>>();
		this.inputIndex = new HashMap<Neuron,Integer>();
		this.outputIndex = new HashMap<Neuron,Integer>();
		
		this.hidden = new ArrayList<ArrayList<Neuron>>();
		this.input = new ArrayList<Neuron>();
 		this.output = new ArrayList<Neuron>();
 		this.learningrate = learningrate;
 		//Input
 		for(int i = 1; i <= input; i++){
 			this.input.add(new Neuron(false));
 		}
 		for(Neuron i : this.input){
 			this.inputIndex.put(i, this.input.indexOf(i));
 		}
 		
 		//Hidden
 		for(int i = 1; i <= numberOfHiddenLayers; i++){
 			ArrayList<Neuron> a = new ArrayList<Neuron>();
 			for(int j = 1; j <= hidden; j++){
 				a.add(new Neuron(true));
 			}
 			this.hidden.add(a);
 		}
 		for(ArrayList<Neuron> a : this.hidden){
 			HashMap<Neuron,Integer> put = new HashMap<Neuron,Integer>();
 			for(Neuron h : a){
 				put.put(h, a.indexOf(h));
 			}
 			this.hiddenIndex.put(this.hidden.indexOf(a), put);
 		}
 		
 		//Output
 		for(int i = 1; i <= output; i++){
 			this.output.add(new Neuron(true));
 		}
 		for(Neuron o : this.output){
 			this.outputIndex.put(o, this.output.indexOf(o));
 		}
 		
 		
 		for(Neuron i : this.input){
 			for(Neuron h : this.hidden.get(0)){
 				i.forbind(h, Math.random()*(Math.random() > 0.5 ? 1 : -1));
 			}
 		}
 		for(int i = 1; i < this.hidden.size(); i++){
 			for(Neuron h : this.hidden.get(i-1)){
 				for(Neuron hto : this.hidden.get(i)){
 					h.forbind(hto, Math.random()*(Math.random() > 0.5 ? 1 : -1));
 				}
 			}
 		}
 		for(Neuron h : this.hidden.get(this.hidden.size()-1)){
 			for(Neuron o : this.output){
 				h.forbind(o, Math.random()*(Math.random() > 0.5 ? 1 : -1));
 			}
 		}
 	}
	
	/**
	 * @param exp The expected value of the input
	 */
	private void backpropagate(double[] exp){
		double[] error = new double[this.output.size()];
		//Hidden->Output
		int c = 0;
		for(Neuron o : this.output){
			error[c] = o.getSenesteOutput()*(1.0-o.getSenesteOutput())*(exp[this.outputIndex.get(o)]-o.getSenesteOutput());
			c++;
		}
		for(Neuron h : this.hidden.get(this.hidden.size()-1)){
			for(Synaps s : h.getForbundetTil()){
				double v = s.getVægt();
				s.setVægt(v+this.learningrate*h.getSenesteOutput()*error[this.outputIndex.get(s.getTil())]);
			}
		}
		double[] oerror = error.clone();
		error = new double[this.hidden.get(0).size()];
		//Hidden->Hidden
		for(int i = this.hidden.size()-1; i > 0; i--){
			c = 0;
			for(Neuron h : this.hidden.get(i)){
				double p = h.getSenesteOutput()*(1-h.getSenesteOutput());
				double k = 0;
				for(Synaps s : h.getForbundetTil()){
					if(i == this.hidden.size()-1){
						k = k+oerror[this.outputIndex.get(s.getTil())]*s.getVægt();
					}
					else{
						k = k+error[this.hiddenIndex.get(i+1).get(s.getTil())]*s.getVægt();
					}
				}
				error[c] = p*k;
				c++;
			}
			for(Neuron h : this.hidden.get(i-1)){
				for(Synaps s : h.getForbundetTil()){
					double v = s.getVægt();
					int index = this.hiddenIndex.get(i).get(s.getTil());
					s.setVægt(v+this.learningrate*error[index]*h.getSenesteInput());
				}
			}
		}
		//Input->Hidden
		c = 0;
		double[] t = error.clone();
		for(Neuron h : this.hidden.get(0)){
			double p = h.getSenesteOutput()*(1.0-h.getSenesteOutput());
			double k = 0;
			for(Synaps s : h.getForbundetTil()){
				if(this.hidden.size() == 1){
					k = k+s.getVægt()*oerror[this.outputIndex.get(s.getTil())];
				}
				else{
					k = k+s.getVægt()*error[this.hiddenIndex.get(1).get(s.getTil())];
				}
			}
			t[c] = k*p;
			c++;
		}
		for(Neuron i : this.input){
			for(Synaps s : i.getForbundetTil()){
				double v = s.getVægt();
				s.setVægt(v+this.learningrate*t[this.hiddenIndex.get(0).get(s.getTil())]*i.getSenesteInput());
			}
		}
	}
	
	private static double activation(double x){
		return 1.0/(1+Math.pow(Math.E, -x));
	}
	
	/**
	 * 
	 * @param input Input to be classified
	 * @return The classification of the input
	 */
	public double[] classify(double[] input) {
		for(int i = 0; i < input.length; i++){
			this.input.get(i).input(input[i]);
		}
		double[] r = new double[this.output.size()];
		for(int i = 0; i < r.length; i++){
			r[i] = this.output.get(i).getSenesteOutput();
		}
		return r;
	}
	
	public double[] map(double[] input){
		for(int i = 0; i < input.length; i++){
			this.input.get(i).input(input[i]);
		}
		double[] retur = new double[this.output.size()];
		for(int i = 0; i < retur.length; i++){
			retur[i] = this.output.get(i).getSenesteOutput();
		}
		return retur;
	}	
	
	public void train(double[] input, double[] gazeLoss){
		for(int i = 0; i < input.length; i++){
			this.input.get(i).input(input[i]);
		}
		backpropagate(gazeLoss);
	}
	
	
	private class Neuron implements Serializable{
		/**
		 * 
		 */
		private static final long serialVersionUID = 6147424346291535969L;
		private boolean act;
		private int antalTriggered = 0, antallinkstil = 0;
		private double senesteinput = 0, senesteoutput = 0, sum;
		private ArrayList<Synaps> forbundetTil;
		public Neuron(boolean act){
			this.act = act;
			this.forbundetTil = new ArrayList<Synaps>();
		}
		
		public void forbind(Neuron e, double vægt){
			Synaps n = new Synaps(e, vægt);
			this.forbundetTil.add(n);
			e.øgLinksTil();
		}
		
		public double getSenesteInput(){
			return this.senesteinput;
		}
		
		public double getSenesteOutput(){
			return this.senesteoutput;
		}
		
		public void input(double input){
			this.antalTriggered++;
			this.sum = sum+input;
			if(this.antalTriggered >= this.antallinkstil){
				this.senesteinput = sum;
				test();
			}
		}
		
		public void test(){
			for(Synaps n : this.forbundetTil){
				if(this.act){
					n.getTil().input(MultiLayerPerceptron.activation(this.sum)*n.getVægt());
				}
				else{
					n.getTil().input(this.sum*n.getVægt());
				}
			}
			if(this.act){
				this.senesteoutput = MultiLayerPerceptron.activation(this.sum);	
			}
			else{
				this.senesteoutput = this.sum;
			}
			this.sum = 0.0;
			this.antalTriggered = 0;
		}
		
		public void øgLinksTil(){
			this.antallinkstil++;
		}
		
		public ArrayList<Synaps> getForbundetTil(){
			return this.forbundetTil;
		}
		
		@Override
		public String toString(){
			String retur = this.hashCode()+" med "+this.forbundetTil.size()+" forbindelser.";
			return retur;
		}
	}
	
	
	/**
	 * 
	 * Private class to connect neurons
	 *
	 */
	private class Synaps implements Serializable{
		/**
		 * 
		 */
		private static final long serialVersionUID = -7099111917741411202L;
		private Neuron til;
		private double vægt;
		public Synaps(Neuron til, double vægt){
			this.til = til;
			this.vægt = vægt;
		}
		
		public double getVægt(){
			return this.vægt;
		}
		
		public void setVægt(double v){
			this.vægt = v;
		}
		
		public Neuron getTil(){
			return this.til;
		}
		
		@Override
		public String toString(){
			return vægt+"";
		}
	}
}