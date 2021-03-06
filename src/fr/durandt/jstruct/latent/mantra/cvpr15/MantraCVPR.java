/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.cvpr15;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.LatentStructuralClassifier;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * Abstract class for MANTRA model used for CVPR 2015 submission
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 * @param <X> pattern
 * @param <Y> output 
 * @param <H> latent
 */
public abstract class MantraCVPR<X,Y,H> implements LatentStructuralClassifier<X,Y,H> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = 6155491818768499925L;

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
	 * Joint feature
	 * @param x
	 * @param y
	 * @param h
	 * @return
	 */
	protected abstract double[] psi(X x, Y y, H h);

	/**
	 * Loss function
	 * @param y is the ground truth label
	 * @param yp
	 * @param hp
	 * @return
	 */
	protected abstract double delta(Y y, Y yp, H hp);

	protected abstract void init(List<STrainingSample<LatentRepresentation<X,H>,Y>> l);

	/**
	 * loss augmented inference
	 * @param ts training sample
	 * @return (yp, hp) <br/>
	 * yp=res[0] : output (Y) <br/>
	 * hp=res[1] : latent (H) <br/>
	 */
	protected abstract Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X,H>,Y> ts, double[] w);

	/**
	 * Compute the latent prediction for a given output y and model w
	 * @param x
	 * @param y
	 * @return argmax_h &lt w, \psi(x,y,h) &gt
	 */
	protected abstract H prediction(X x, Y y, double[] w);

	/**
	 * Define how the model is learned
	 * @param l
	 */
	protected abstract void learning(List<STrainingSample<LatentRepresentation<X,H>,Y>> l);

	/**
	 * Compute the best output and latent values for a given model w
	 * @param x
	 * @param w
	 * @return
	 */
	protected abstract Object[] predictionOutputLatent(X x, double[] w);

	/**
	 * argmax_y { max_h ( &lt w, psi(x,y,h) &gt ) - max_y'!=y min_h &lt w, psi(x,y,h) &gt <br/>
	 * @param x
	 * @param w
	 * @return
	 */
	protected abstract Y prediction(LatentRepresentation<X, H> x, double[] w);

	/**
	 * 
	 * @param x
	 * @param y
	 * @return (hp,ym,hm)
	 *  hp = argmax_h &lt w, psi(x,y,h) &gt <br/>
	 *  ym = argmax_{y'!=y} { min_h &lt w, psi(x,y',h) &gt } <br/>
	 *  hm = argmin_h &lt w, psi(x,ym,h) &gt <br/>
	 */
	protected abstract Object[] inferPrediction(X x, Y y, double[] w);


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	@Override
	public void train(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
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
	 * Compute &lt w, Psi(x,y,h) &gt
	 * @param x
	 * @param y
	 * @param h
	 * @return  &lt w, Psi(x,y,h) &gt
	 */
	protected double valueOf(X x, Y y, H h, double[] w) {
		return linear.valueOf(w, psi(x, y, h));
	}

	/**
	 * Compute &lt w, Psi(x,y,h) &gt
	 * @param x
	 * @param y
	 * @param h
	 * @return  &lt w, Psi(x,y,h) &gt
	 */
	protected double valueOf(X x, Y y, H h) {
		return valueOf(x, y, h, w);
	}

	/**
	 * Compute the loss of the objective function <br />
	 * \sum_{i=1}^N max_{y,h} ( delta(yi,y) + &lt w, \psi(xi,y,h) &gt ) - max_hp &lt w,psi(xi,yi,hp) &gt
	 * @param l list of training samples
	 */
	protected double loss(final List<STrainingSample<LatentRepresentation<X,H>,Y>> l, final double[] w) {
		double loss = 0;

		if(nThreads > 1) { // Multi-threads execution

			ExecutorService executor = Executors.newFixedThreadPool(nThreads);
			List<Future<Double>> futures = new ArrayList<Future<Double>>();
			CompletionService<Double> completionService = new ExecutorCompletionService<Double>(executor);

			for(int i=0 ; i<l.size(); i++) {
				final int n = i;
				futures.add(completionService.submit(new Callable<Double>() {

					@Override
					public Double call() throws Exception {
						STrainingSample<LatentRepresentation<X,H>,Y> ts = l.get(n);
						// Compute the loss augmented inference
						Object[] or = lossAugmentedInference(ts, w);
						Y yp = (Y)or[0];
						H hp = (H)or[1];
						Y ym = (Y)or[2];
						H hm = (H)or[3];

						// Compute the "best" latent value for ground truth output
						Object[] res = inferPrediction(ts.input.x, ts.output, w);
						H hpi = (H)res[0];
						Y ymi = (Y)res[1];
						H hmi = (H)res[2];
						// Compute the loss with the predicted output and latent variables
						return delta(ts.output, yp, hp) + valueOf(ts.input.x, yp, hp) - valueOf(ts.input.x, ym, hm)
								- valueOf(ts.input.x, ts.output, hpi) + valueOf(ts.input.x, ymi, hmi);
					}
				}));
			}

			for(Future<Double> f : futures) {
				try {
					Double res = f.get();
					if (res != null) {
						loss += res;
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			executor.shutdown();
		}
		else { // Mono-thread execution
			for(STrainingSample<LatentRepresentation<X,H>,Y> ts : l) {
				// Compute the loss augmented inference
				Object[] or = lossAugmentedInference(ts, w);
				Y yp = (Y)or[0];
				H hp = (H)or[1];
				Y ym = (Y)or[2];
				H hm = (H)or[3];

				// Compute the "best" latent value for ground truth output
				Object[] res = inferPrediction(ts.input.x, ts.output, w);
				H hpi = (H)res[0];
				Y ymi = (Y)res[1];
				H hmi = (H)res[2];
				// Compute the loss with the predicted output and latent variables
				loss += delta(ts.output, yp, hp) + valueOf(ts.input.x, yp, hp) - valueOf(ts.input.x, ym, hm)
						- valueOf(ts.input.x, ts.output, hpi) + valueOf(ts.input.x, ymi, hmi);
			}
		}
		loss /= l.size();
		return loss;	
	}

	/**
	 * Compute the primal objective value for the given model w
	 * @param l list of training samples
	 * @return primal objective value
	 */
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,Y>> l, double[] w) {
		double obj = lambda * VectorOperations.dot(w,w)/2;
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
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		return primalObj(l, w);
	}

	/**
	 * Compute the empirical risk <br />
	 * @parm l list of samples
	 * @return \sum_{i=1}^N delta(yi,y) <br/>
	 * where (y,h) = argmax_{y,h} &lt w, \psi(xi,y,h) &gt
	 */
	protected double empiricalRisk(final List<STrainingSample<LatentRepresentation<X,H>,Y>> l, final double[] w) {
		double risk = 0;

		if(nThreads > 1) { // Multi-threads execution
			ExecutorService executor = Executors.newFixedThreadPool(nThreads);
			List<Future<Double>> futures = new ArrayList<Future<Double>>();
			CompletionService<Double> completionService = new ExecutorCompletionService<Double>(executor);

			for(int i=0 ; i<l.size(); i++) {
				final int n = i;
				futures.add(completionService.submit(new Callable<Double>() {

					@Override
					public Double call() throws Exception {
						STrainingSample<LatentRepresentation<X,H>,Y> ts = l.get(n);
						// Compute the "best" output and latent values
						Object[] or = predictionOutputLatent(ts.input.x);
						Y yp = (Y)or[0];
						H hp = (H)or[1];
						// Compute the loss with the latent and output prediction
						return delta(ts.output, yp, hp);
					}
				}));
			}

			for(Future<Double> f : futures) {
				try {
					Double res = f.get();
					if (res != null) {
						risk += res;
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			executor.shutdown();
		}
		else { // Mono-thread execution
			for(STrainingSample<LatentRepresentation<X,H>,Y> ts : l) {
				// Compute the "best" output and latent values
				Object[] or = predictionOutputLatent(ts.input.x);
				Y yp = (Y)or[0];
				H hp = (H)or[1];
				// Compute the loss with the latent and output prediction
				risk += delta(ts.output, yp, hp);
			}
		}
		risk /= l.size();
		return risk;
	}

	public double empiricalRisk(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		return empiricalRisk(l, w);
	}

	/**
	 * Compute the latent prediction for given output y
	 * @param x
	 * @param y
	 * @return
	 */
	protected H prediction(X x, Y y) {
		return prediction(x, y, w);
	}

	@Override
	public Y prediction(LatentRepresentation<X,H> x) {
		return prediction(x,w);
	}

	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X,H>,Y> ts) {
		return lossAugmentedInference(ts, w);
	}

	@Override
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
