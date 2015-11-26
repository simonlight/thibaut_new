/**
 * 
 */
package lsvm;

import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LSVM<X,H> implements Classifier<LatentRepresentation<X, H>> {

	protected double lambda = 1e-4;

	protected int verbose = 0;
	
	protected int[] nb = new int[2];

	//svm hyperplane
	protected double[] w = null;
	protected int dim = 0;

	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();

	// abstract methods
	protected abstract double[] psi(X x, H h);
	/**
	 * Initialize latent variables
	 * @param l
	 */
	protected abstract void init(List<TrainingSample<LatentRepresentation<X,H>>> l);
	protected abstract void learn(List<TrainingSample<LatentRepresentation<X,H>>> l);
	protected abstract H optimizeH(X x);


	@Override
	public double valueOf(LatentRepresentation<X,H> rep) {
		H hp = optimizeH(rep.x);
		return linear.valueOf(w, psi(rep.x, hp));
	}
	
	public double valueOf(X x, H h) {
		return linear.valueOf(w, psi(x, h));
	}

	@Override
	public void train(TrainingSample<LatentRepresentation<X,H>> t) {
		// TODO Auto-generated method stub

	}

	@Override
	public void train(List<TrainingSample<LatentRepresentation<X,H>>> l) {

		if(l.isEmpty())
			return;

		for(TrainingSample<LatentRepresentation<X,H>> ts : l) {
			if(ts.label == 1) {
				nb[0] += 1;
			}
			else if(ts.label == -1) {
				nb[1] += 1;
			}
			else {
				System.out.println("ERROR: label is not +1/-1 " + ts.label);
				System.exit(0);
			}
		}

		// initialize latent variables and dim
		init(l);

		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train latent SVM \tw: " + dim + "\tlambda: " + lambda); 
		System.out.println("nb= " + l.size() + "\t" + Arrays.toString(nb));
		showParameters();
		System.out.println("----------------------------------------------------------------------------------------");

		// initialize w
		w = new double[dim];

		// Train LSVM 
		long startTime = System.currentTimeMillis();
		learn(l);
		long endTime = System.currentTimeMillis();
		System.out.println("END LEARNING - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("primal obj= " + getPrimalObjective(l));

		// compute accuracy
		accuracy(l);

	}

	protected void optimizeLatentPos(List<TrainingSample<LatentRepresentation<X,H>>> l){
		for(TrainingSample<LatentRepresentation<X,H>> ts : l){
			if(ts.label == 1){
				ts.sample.h = optimizeH(ts.sample.x);
			}
		}
	}

	protected void optimizeLatentNeg(List<TrainingSample<LatentRepresentation<X,H>>> l){
		for(TrainingSample<LatentRepresentation<X,H>> ts : l){
			if(ts.label == -1){
				ts.sample.h = optimizeH(ts.sample.x);
			}
		}
	}

	protected void optimizeAllLatent(List<TrainingSample<LatentRepresentation<X,H>>> l){
		for(TrainingSample<LatentRepresentation<X,H>> ts : l){
			ts.sample.h = optimizeH(ts.sample.x);
		}
	}

	protected double accuracy(List<TrainingSample<LatentRepresentation<X,H>>> l){
		double accuracy = 0;
		int nb = 0;
		for(TrainingSample<LatentRepresentation<X,H>> ts : l){
			double fxiyi = ts.label * valueOf(ts.sample);
			if(fxiyi>0){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}

	/**
	 * Compute the loss function for the list of training sample l
	 * @param l
	 * @return the loss function
	 */
	public double getLoss(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		double loss = 0;
		for(TrainingSample<LatentRepresentation<X,H>> ts : l) {
			loss += loss(ts);
		}
		loss /= l.size();
		return loss;
	}

	/**
	 * Compute the loss function for one sample ts
	 * @param ts
	 * @return
	 */
	public double loss(TrainingSample<LatentRepresentation<X,H>> ts) {
		double v = valueOf(ts.sample.x, ts.sample.h);
		return Math.max(0, 1 - ts.label * v);
	}

	/** 
	 * Compute the primal objective
	 * @param l
	 * @return the primal objective
	 */
	public double getPrimalObjective(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		double obj = 0;
		obj += VectorOperations.dot(w,w) * lambda/2;
		double loss = getLoss(l);
		obj += loss;
		return obj;
	}

	protected void showParameters() {}
	
	
	/**
	 * @return the lambda
	 */
	public double getLambda() {
		return lambda;
	}
	/**
	 * @param lambda the lambda to set
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	/**
	 * @return the verbose
	 */
	public int getVerbose() {
		return verbose;
	}
	/**
	 * @param verbose the verbose to set
	 */
	public void setVerbose(int verbose) {
		this.verbose = verbose;
	};
	
	

}
