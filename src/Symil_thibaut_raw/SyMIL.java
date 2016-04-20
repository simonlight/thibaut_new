package Symil_thibaut_raw;
/**
 * 
 */


import java.util.ArrayList;
import java.util.List;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class SyMIL<X,H> implements Classifier<LatentRepresentation<X, H>> {

	protected double lambda = 1e-4;
	
	/**
	 * trade-off
	 */
	protected double gamma = 1;

	protected int verbose = 0;

	protected int nbPlus = 0;
	protected int nbMinus = 0;

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
	protected abstract void init(List<TrainingSample<LatentRepresentation<X,H[]>>> l);
	protected abstract void learn(List<TrainingSample<LatentRepresentation<X,H[]>>> l);
	/**
	 * 
	 * @param x
	 * @return the max and the min
	 */
	protected abstract H[] optimizeH(X x);

	@Override
	public double valueOf(LatentRepresentation<X,H> rep) {
		// compute h^+ and h^-
		H[] hpredict = optimizeH(rep.x);
		// compute the score for h^+
		double vp = valueOf(rep.x, hpredict[0]);
		// compute the score for h^-
		double vm = valueOf(rep.x, hpredict[1]);
		if(vp > -vm) {
			// return the score for h^+
			return vp;
		}
		// return the score for h^-
		return vm;
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

		nbPlus = 0;
		nbMinus = 0;
		for(TrainingSample<LatentRepresentation<X,H>> ts : l) {
			if(ts.label == 1) {
				nbPlus += 1;
			}
			else if(ts.label == -1) {
				nbMinus += 1;
			}
			else {
				System.out.println("ERROR: label is not +1/-1 " + ts.label);
				System.exit(0);
			}
		}

		List<TrainingSample<LatentRepresentation<X,H[]>>> lt = new ArrayList<TrainingSample<LatentRepresentation<X,H[]>>>();
		for(int i=0; i<l.size(); i++) {
			lt.add(new TrainingSample<LatentRepresentation<X,H[]>>(new LatentRepresentation<X,H[]>(l.get(i).sample.x, (H[])new Object[2]),l.get(i).label));
		}

		// initialize latent variables and dim
		init(lt);

		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train SyMIL \tw: " + dim + "\tlambda= " + lambda + "\tgamma= " + gamma); 
		System.out.println("nb= " + l.size() + "\tn^+= " + nbPlus + "\tn^-= " + nbMinus);
		showParameters();
		System.out.println("----------------------------------------------------------------------------------------");

		nbPlus = 1;
		nbMinus = 1;
		
		// initialize w
		w = new double[dim];

		// Train 
		long startTime = System.currentTimeMillis();
		learn(lt);
		long endTime = System.currentTimeMillis();
		System.out.println("END LEARNING - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("primal obj= " + getPrimalObjective(l));

		// compute accuracy
		accuracy(l);

	}

	public double accuracy(List<TrainingSample<LatentRepresentation<X,H>>> l){
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
		double loss = 0;
		// compute h^+ and h^-
		H[] hpredict = optimizeH(ts.sample.x);
		// compute the score for h^+
		double vp = valueOf(ts.sample.x, hpredict[0]);
		// compute the score for h^-
		double vm = valueOf(ts.sample.x, hpredict[1]);

		if(ts.label == 1) {
			loss += Math.max(0, 1 - ts.label * vp)/nbPlus;
		}
		else if(ts.label == -1) {
			loss += Math.max(0, 1 - ts.label * vm)/nbMinus;
		}
		loss += gamma * Math.max(0, 1 - ts.label * (vp + vm)) / (nbPlus + nbMinus);

		return loss;
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

	protected void optimizeAllLatent(List<TrainingSample<LatentRepresentation<X,H[]>>> l){
		for(TrainingSample<LatentRepresentation<X,H[]>> ts : l){
			ts.sample.h = optimizeH(ts.sample.x);
		}
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
	}
	/**
	 * @return the gamma
	 */
	public double getGamma() {
		return gamma;
	}
	/**
	 * @param gamma the gamma to set
	 */
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}


}
