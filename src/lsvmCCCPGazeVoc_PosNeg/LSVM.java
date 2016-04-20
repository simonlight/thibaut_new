/**
 * 
 */
package lsvmCCCPGazeVoc_PosNeg;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
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
public abstract class LSVM<X,H> implements Classifier<LatentRepresentation<X, H>> , Serializable  {

	/**
	 * 
	 */
	private static final long serialVersionUID = -742534151257155346L;

	/**
	 * 
	 */

	protected double lambda;

	protected int verbose;
	
	protected int[] nb = new int[2];

	//svm hyperplane
	protected double[] w = null;
	protected int dim = 0;

	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();
//	protected BufferedWriter trainingDetailFileOut;

	// abstract methods
	protected abstract double[] psi(X x, H h);
	/**
	 * Initialize latent variables
	 * @param l
	 */
	protected abstract void init(List<TrainingSample<LatentRepresentation<X,H>>> l);
	protected abstract void learn(List<TrainingSample<LatentRepresentation<X,H>>> l);
	protected abstract void learn(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut);
	protected abstract H optimizeH(X x);
	protected abstract double[] loss(TrainingSample<LatentRepresentation<X,H>> ts);
	public abstract double getLoss(List<TrainingSample<LatentRepresentation<X,H>>> l);
	public abstract double getLoss(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut);

//	@Override
//	public double valueOf(LatentRepresentation<X,H> rep) {
//		H hp = optimizeH(rep.x);
//		return linear.valueOf(w, psi(rep.x, hp));
//	}
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
	public void train(List<TrainingSample<LatentRepresentation<X,H>>> l){
		
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

		// initialize w
		w = new double[dim];

		// Train LSVM 
		long startTime = System.currentTimeMillis();
		learn(l);
		long endTime = System.currentTimeMillis();
		System.out.println("END LEARNING - Time learning= "+ (endTime-startTime)/1000 + "s");
//		System.out.println("primal obj= " + getPrimalObjective(l));

		// compute accuracy
		accuracy(l);

	}
	public void train(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut){
		
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

		// initialize w
		w = new double[dim];

		// Train LSVM 
		long startTime = System.currentTimeMillis();
		learn(l, trainingDetailFileOut);
		long endTime = System.currentTimeMillis();
		System.out.println("END LEARNING - Time learning= "+ (endTime-startTime)/1000 + "s");
//		System.out.println("primal obj= " + getPrimalObjective(l));
		try {
			trainingDetailFileOut.write(" time_learning:"+(endTime-startTime)/1000 + "s");
			trainingDetailFileOut.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// compute accuracy
		accuracy(l);

	}
	
	protected void optimizeLatent(List<TrainingSample<LatentRepresentation<X,H>>> l){
		for(TrainingSample<LatentRepresentation<X,H>> ts : l){
			ts.sample.h = optimizeH(ts.sample.x);
		}
	}
	
	protected void optimizePositiveLatent(List<TrainingSample<LatentRepresentation<X,H>>> l){
		for(TrainingSample<LatentRepresentation<X,H>> ts : l){
			if (ts.label==1){
				ts.sample.h = optimizeH(ts.sample.x);
			}
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


	public double getPrimalObjective(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		double obj = 0;
		obj += VectorOperations.dot(w,w) * lambda/2;
		double loss = getLoss(l);
		obj += loss;
		return obj;
	}
	
	public double getPrimalObjective(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut) {
		double obj = 0;
		obj += VectorOperations.dot(w,w) * lambda/2;
		double loss = getLoss(l ,trainingDetailFileOut);
		obj += loss;
		try {
			trainingDetailFileOut.write(" objectif:"+obj+"\n");
			trainingDetailFileOut.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
