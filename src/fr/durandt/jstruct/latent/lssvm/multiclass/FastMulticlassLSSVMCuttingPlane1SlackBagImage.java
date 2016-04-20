/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class FastMulticlassLSSVMCuttingPlane1SlackBagImage extends FastMulticlassLSSVMCuttingPlane1Slack<BagImage,Integer> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = 3417634151825173686L;
	

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#psi(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImage x, Integer h) {
		return x.getInstance(h);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#delta(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(Integer y, Integer yp, Integer hp) {
		if(y == yp) {
			return 0;
		}
		else {
			return 1;
		}
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagImage, Integer>, Integer>> l) {
		
		// Count the number of classes
		int nbClass = 0;	// Number of classes
		for(STrainingSample<LatentRepresentation<BagImage, Integer>, Integer> ts : l) {
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
		for(STrainingSample<LatentRepresentation<BagImage, Integer>, Integer> ts : l) {
			nb[ts.output]++;
		}

		// Print the classes and the number of samples per class
		System.out.println("Multiclass SSVM - classes: " + listClass + "\t" + Arrays.toString(nb));

		// Define the dimension of w
		dim = l.get(0).input.x.getInstance(0).length;

		// Initialize w
		w = new double[listClass.size()][dim];

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample)
	 */
	@Override
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<BagImage, Integer>, Integer> ts, double[][] w) {
		int ypredict = -1;	// class prediction
		Integer hpredict = null;	// latent prediction
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {	// For each class
			for(int h=0; h<ts.input.x.numberOfInstances(); h++) {	// For each latent
				double val = delta(ts.output, y, h) + valueOf(ts.input.x, y, h, w);
				if(val>valmax){
					valmax = val;
					ypredict = y;
					hpredict = h;
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected Integer prediction(BagImage x, Integer y, double[][] w) {
		double max = -Double.MAX_VALUE;
		int hpredict = -1; // Latent prediction
		for(int h=0; h<x.numberOfInstances(); h++) {	// For each region
			// Compute the score of region h
			double score = valueOf(x, y, h, w);
			if(score > max) {
				max = score;
				hpredict = h;
			}
		}
		return hpredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.LatentStructuralClassifier#predictionOutputLatent(java.lang.Object)
	 */
	@Override
	protected Object[] predictionOutputLatent(BagImage x, double[][] w) {
		int ypredict = -1;	// class prediction
		Integer hpredict = null;	// latent prediction
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {	// For each class
			for(int h=0; h<x.numberOfInstances(); h++) {	// For each latent
				// Compute the score for a given class y and region h
				double val = valueOf(x, y, h, w);
				if(val>valmax){
					valmax = val;
					ypredict = y;
					hpredict = h;
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.struct.StructuralClassifier#prediction(java.lang.Object)
	 */
	@Override
	protected Integer prediction(LatentRepresentation<BagImage, Integer> x, double[][] w) {
		Object[] or = predictionOutputLatent(x.x, w);
		return (Integer)or[0];
	}

	/**
	 * Compute the multiclass accuracy
	 * @param l
	 * @return
	 */
	public double accuracy(List<STrainingSample<LatentRepresentation<BagImage, Integer>, Integer>> l){
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ts : l){
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
	public String toString() {
		return "fast_multiclass_" + super.toString();
	}
}
