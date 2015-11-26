/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.LatentStructuralClassifier;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class RankingMantraM2<X,H> implements LatentStructuralClassifier<X,RankingOutput,H> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1016383442866409895L;

	/**
	 * regularisation parameter
	 */
	protected double lambda = 1e-4;

	/**
	 * weight vector
	 */
	protected double[] w = null;

	/**
	 * dimension of the weight vector
	 */
	protected int dim = 0;

	/**
	 * linear kernel
	 */
	protected DoubleLinear linear = new DoubleLinear();

	/**
	 * Time used to compute the loss augmented inference
	 */
	protected int lossAugmentedTime = 0;

	/**
	 * number of threads
	 */
	protected int nThreads = 1;

	protected int verbose = 0;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Abstract methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Loss function
	 * @param y
	 * @param yp
	 * @param hp
	 * @return
	 */
	protected abstract double delta(RankingOutput y, RankingOutput yp, H hp);

	protected abstract void init(List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l);

	/**
	 * Compute the latent prediction for a given output y and model w
	 * @param x
	 * @param y
	 * @return argmax_h &lt w, \psi(x,y,h) &gt
	 */
	protected abstract H prediction(X x, RankingOutput y, double[] w);

	/**
	 * Define how the model is learned
	 * @param l
	 */
	protected abstract void learning(List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l);

	/**
	 * Compute the best output and latent values for a given model w
	 * @param x
	 * @param w
	 * @return
	 */
	protected abstract Object[] predictionOutputLatent(X x, double[] w);

	protected abstract RankingOutput prediction(LatentRepresentation<X, H> x, double[] w);

	/**
	 * loss augmented inference <br/>
	 * y = argmax_y max_hp &lt w, psi(x,y,hp) &gt + min_hm &lt w, psi(x,y,hm) &gt
	 * @param ts training sample
	 * @return (y, hp, hm) <br/>
	 * y=res[0] : output (Y) <br/>
	 * hp=res[1] : latent (H) <br/>
	 * hm=res[2] : latent (H) <br/>
	 */
	protected abstract Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X,H>,RankingOutput> ts, double[] w);
	
	/**
	 * Compute &lt w, Psi(x,y,h) &gt + min_hm &lt w, Psi(x,y,hm) &gt
	 * @param x
	 * @param y
	 * @param h
	 * @return &lt w, Psi(x,y,h) &gt + min_hm &lt w, Psi(x,y,hm) &gt
	 */
	protected abstract double valueOf(X x, RankingOutput y, H h);


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	public void train(List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l) {
		if(l.isEmpty())
			return;

		System.out.println("----------------------------------------------------------------------------------------");
		// Initialization
		init(l);
		System.out.println("----------------------------------------------------------------------------------------");
		// Print the parameters
		showParameters();
		System.out.println("----------------------------------------------------------------------------------------");

		long startTime = System.currentTimeMillis();
		lossAugmentedTime = 0;
		// Lean the model w
		learning(l);
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim - Time learning= "+ (endTime-startTime)/1000 + "s");
		// Print the primal objective
		System.out.println("primal obj= " + primalObj(l));
		// Compute the empirical risk
		double risk = empiricalRisk(l);
		System.out.println("Empirical risk= " + risk);

		System.out.println("----------------------------------------------------------------------------------------");
	}

	/**
	 * Compute the loss of the objective function <br />
	 * \sum_{i=1}^N max_{y,h} ( delta(yi,y) + &lt w, \psi(xi,y,h) &gt ) - max_hp &lt w,psi(xi,yi,hp) &gt
	 * @param l list of training samples
	 */
	protected double loss(final List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l, final double[] w) {
		double loss = 0;
		for(STrainingSample<LatentRepresentation<X,H>,RankingOutput> ts : l) {
			// Compute the loss augmented inference
			Object[] or = lossAugmentedInference(ts, w);
			RankingOutput yp = (RankingOutput)or[0];
			H hp = (H)or[1];
			// Compute the "best" latent value for ground truth output
			H h = prediction(ts.input.x, ts.output, w);
			// Compute the loss with the predicted output and latent variables
			loss += delta(ts.output, yp, hp) + valueOf(ts.input.x, yp, hp) - valueOf(ts.input.x, ts.output, h);
		}
		loss /= l.size();
		return loss;	
	}

	/**
	 * Compute the primal objective value for the given model w
	 * @param l list of training samples
	 * @return primal objective value
	 */
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l, double[] w) {
		double obj = lambda * VectorOp.dot(w,w)/2;
		// Compute the loss of the objective function
		double loss = loss(l, w);
		System.out.println("lambda*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}

	/**
	 * Compute the primal objective value for the current model w
	 * @param l list of training samples
	 * @return primal objective value
	 */
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l) {
		return primalObj(l,w);
	}

	/**
	 * Compute the empirical risk <br />
	 * @parm l list of samples
	 * @return \sum_{i=1}^N delta(yi,y) <br/>
	 * where (y,h) = argmax_{y,h} &lt w, \psi(xi,y,h) &gt
	 */
	protected double empiricalRisk(final List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l, final double[] w) {
		double risk = 0;
		for(STrainingSample<LatentRepresentation<X,H>,RankingOutput> ts : l) {
			// Compute the "best" output and latent values
			Object[] or = predictionOutputLatent(ts.input.x);
			RankingOutput yp = (RankingOutput)or[0];
			H hp = (H)or[1];
			// Compute the loss with the latent and output prediction
			risk += delta(ts.output, yp, hp);
		}
		risk /= l.size();
		return risk;
	}

	public double empiricalRisk(final List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l) {
		return empiricalRisk(l, w);
	}

	/**
	 * Compute the latent prediction for given output y
	 * @param x
	 * @param y
	 * @return
	 */
	protected H prediction(X x, RankingOutput y) {
		return prediction(x, y, w);
	}

	public RankingOutput prediction(LatentRepresentation<X,H> x) {
		return prediction(x,w);
	}

	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X,H>,RankingOutput> ts) {
		return lossAugmentedInference(ts, w);
	}

	public Object[] predictionOutputLatent(X x) {
		return predictionOutputLatent(x, w);
	}

	/**
	 * Print the parameters
	 */
	protected void showParameters(){
		System.out.println("Train MANTRA \tlambda: " + lambda + "\tdim: " + dim + "\tthreads= " + nThreads);
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

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
	 * @return the w
	 */
	public double[] getW() {
		return w;
	}

	/**
	 * @param w the w to set
	 */
	public void setW(double[] w) {
		this.w = w;
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
	 * @return the nThreads
	 */
	public int getnThreads() {
		return nThreads;
	}

	/**
	 * @param nThreads the nThreads to set
	 */
	public void setnThreads(int nThreads) {
		int nThreadsMax = Runtime.getRuntime().availableProcessors();
		this.nThreads = Math.min(nThreadsMax, nThreads);
	}

}
