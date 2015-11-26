package fr.durandt.jstruct.ssvm.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;

/**
 * Structural SVM for multiclass classification
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class DoubleFastMulticlassSSVMFrankWolfe extends FastMulticlassSSVMFrankWolfe<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4539542482619545852L;

	@Override
	protected Integer prediction(double[] x, double[][] w) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			double val = valueOf(x, y, w);
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	@Override
	protected Integer lossAugmentedInference(STrainingSample<double[], Integer> ts, double[][] w) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			double val = delta(ts.output, y) + valueOf(ts.input, y, w);
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		return ypredict;
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
		return x;
	}

	@Override
	protected void init(List<STrainingSample<double[], Integer>> l) {

		int nbClass = 0;
		for(STrainingSample<double[], Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;

		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		double[] nb = new double[nbClass];
		for(STrainingSample<double[], Integer> ts : l) {
			nb[ts.output]++;
		}
		// print the classes and the number of samples per class
		System.out.println("Fast Multiclass SSVM - classes: " + listClass + "\t" + Arrays.toString(nb));

		// define the dimension of w
		dim = l.get(0).input.length;

		// initialize the dimension of w
		w = new double[listClass.size()][dim];
	}

	public String toString() {
		return super.toString();
	}
}
