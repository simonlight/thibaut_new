package fr.durandt.jstruct.ssvm.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.ssvm.SSVMPegasos;
import fr.durandt.jstruct.struct.STrainingSample;

/**
 * Structural SVM for multiclass classification.
 * The optimisation problem is solved with Pegasos
 * 
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class DoubleMulticlassSSVMPegasos extends SSVMPegasos<double[], Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1536760878851523970L;
	
	protected List<Integer> listClass = null;

	@Override
	protected Integer prediction(double[] x, double[] w) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			double val = valueOf(x,y,w);
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	@Override
	protected Integer lossAugmentedInference(STrainingSample<double[], Integer> ts, double[] w) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			double val = delta(ts.output,y) + valueOf(ts.input,y,w);
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	public double multiclassAccuracy(List<STrainingSample<double[], Integer>> l) {
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<double[], Integer> ts : l){
			int ypredict = prediction(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}

	@Override
	protected double delta(Integer yi, Integer y) {
		if(y == yi) {
			return 0;
		}
		else {
			return 1;
		}
	}

	@Override
	protected double[] psi(double[] x, Integer y) {
		double[] psi = new double[dim];
		for(int i=0; i<x.length; i++) {
			psi[y*x.length+i] = x[i];
		}
		return psi;
	}

	@Override
	protected void init(List<STrainingSample<double[], Integer>> l) {

		// Count the number of classes
		int nbClass = 0;	// Number of classes
		for(STrainingSample<double[], Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;

		// Create the list of classes
		listClass = new ArrayList<Integer>();	// List of classes
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		// Count the number of samples per class
		double[] nb = new double[nbClass];
		for(STrainingSample<double[], Integer> ts : l) {
			nb[ts.output]++;
		}
		
		// Print the classes and the number of samples per class
		System.out.println("Multiclass SSVM - classes: " + listClass + "\t" + Arrays.toString(nb));

		// Define the dimension of w
		dim = l.get(0).input.length*listClass.size();
		
		// Initialize w
		w = new double[dim];
	}

	@Override
	public String toString() {
		return "multiclass_" + super.toString();
	}
}
