/**
 * 
 */
package Symil_Musk;

import java.io.Serializable;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentationSymil;
import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LSVM<X,H> implements Classifier<LatentRepresentationSymil<X, H, H>> , Serializable  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1668592777976093022L;

	protected double lambda;

	protected int verbose;
	
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

	protected abstract void init(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l);
	protected abstract void learn(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l);
	protected abstract H[] optimizeH(X x);
	protected abstract H optimizePositiveH(X x);
	protected abstract H optimizeNegativeH(X x);
	protected abstract double[] loss(TrainingSample<LatentRepresentationSymil<X, H, H>> ts);
	public abstract double getLoss(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l);

	@Override
	public double valueOf(LatentRepresentationSymil<X, H, H> rep) {
		return null;
	}
	
	public double valueOf(X x, H h) {
		return linear.valueOf(w, psi(x, h));
	}

	@Override
	public void train(TrainingSample<LatentRepresentationSymil<X, H, H>> t) {
		// TODO Auto-generated method stub

	}	

@Override
public void train(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l){
		
		if(l.isEmpty())
			return;
		
		for(TrainingSample<LatentRepresentationSymil<X, H, H>> ts : l) {
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

protected void optimizeLatent(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l){
	for(TrainingSample<LatentRepresentationSymil<X, H, H>> ts : l){
			H[] hphn = optimizeH(ts.sample.x);
			ts.sample.hp =hphn[0];
			ts.sample.hn =hphn[1];
		}
}
	
protected double accuracy(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l){
	double accuracy = 0;
	int nb = 0;
	int cnt=0;
	for(TrainingSample<LatentRepresentationSymil<X, H, H>> ts : l){
		double fxiyi = ts.label * ( valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn) );
//		System.out.println("i="+cnt+" prediction="+ (valueOf(ts.sample.x, ts.sample.hp) + valueOf(ts.sample.x, ts.sample.hn)>0?1:-1) + " gt:"+ts.label);
		cnt+=1;
		if(fxiyi>0){	
			nb++;
		}
	}
	accuracy = (double)nb/(double)l.size();
	System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
	return accuracy;
}


	public double getPrimalObjective(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l) {
		double obj = 0;
		obj += VectorOperations.dot(w,w) *lambda /2;
		double loss = getLoss (l);
		obj += loss;
		System.out.println("obj:"+obj);

		return obj;
	}

	protected void showParameters() {}
	
	
	public double getLambda() {
		return lambda;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	public int getVerbose() {
		return verbose;
	}
	public void setVerbose(int verbose) {
		this.verbose = verbose;
	};
	
	

}
