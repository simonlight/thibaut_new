package fr.durandt.jstruct.ssvm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.struct.StructuralClassifier;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;

/**
 * Abstract class to solve SSVM.
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 * @param <X> pattern
 * @param <Y> output
 */
public abstract class SSVM<X,Y> implements StructuralClassifier<X, Y> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = -712338181556012795L;

	/**
	 * Regularization parameter
	 */
	protected double lambda = 1e-4;

	protected int verbose = 0;

	/**
	 * Time used to compute the loss augmented inference
	 */
	protected int lossAugmentedTime = 0;

	/**
	 * svm hyperplane
	 */
	protected double[] w = null;

	/**
	 *  dimension of w
	 */
	protected int dim = 0;	

	/**
	 * linear kernel
	 */
	protected DoubleLinear linear = new DoubleLinear();

	/**
	 * number of threads
	 */
	protected int nThreads = 1;

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Abstract methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	protected abstract void learning(List<STrainingSample<X, Y>> l);
	protected abstract Y lossAugmentedInference(STrainingSample<X, Y> ts, double[] w);
	protected abstract Y prediction(X x, double[] w);
	protected abstract double delta(Y yi, Y y);
	protected abstract double[] psi(X x, Y y);

	/**
	 * This method must: <br />
	 * - define the dimension of w <br />
	 * - initialize w
	 * @param l
	 */
	protected abstract void init(List<STrainingSample<X, Y>> l);

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	@Override
	public void train(List<STrainingSample<X, Y>> l) {

		if(l.isEmpty())
			return;

		System.out.println("----------------------------------------------------------------------------------------");
		init(l);
		System.out.println("----------------------------------------------------------------------------------------");
		showParameters();
		System.out.println("----------------------------------------------------------------------------------------");

		long startTime = System.currentTimeMillis();
		lossAugmentedTime = 0;
		learning(l);
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("primal obj= " + primalObj(l));
		double risk = empiricalRisk(l);
		System.out.println("Empirical risk= " + risk);

		System.out.println("----------------------------------------------------------------------------------------");
	}

	/**
	 * Compute the loss function for a given w
	 * @param l
	 * @param w
	 * @return the loss function
	 */
	protected double loss(final List<STrainingSample<X, Y>> l, final double[] w) {
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
						STrainingSample<X, Y> ts = l.get(n);
						Y yp = lossAugmentedInference(ts, w);
						return delta(ts.output, yp) + valueOf(ts.input, yp, w) - valueOf(ts.input, ts.output, w);
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
			for(STrainingSample<X, Y> ts : l) {
				Y yp = lossAugmentedInference(ts, w);
				loss += delta(ts.output, yp) + valueOf(ts.input, yp, w) - valueOf(ts.input, ts.output, w);
			}
		}
		loss /= l.size();
		return loss;	
	}

	/**
	 * Compute the loss function
	 * @param l
	 * @return
	 */
	protected double loss(List<STrainingSample<X, Y>> l) {
		return loss(l, w);
	}

	/**
	 * Compute the primal objective for given w
	 * @param l
	 * @param w
	 * @return
	 */
	protected double primalObj(List<STrainingSample<X, Y>> l, double[] w) {
		double obj = lambda * linear.valueOf(w,w)/2;
		double loss = loss(l,w);
		System.out.println("lambda/2*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}

	/**
	 * Compute the primal objective
	 * @param l
	 * @return
	 */
	public double primalObj(List<STrainingSample<X, Y>> l) {
		return primalObj(l, w);
	}

	/**
	 * Compute the empirical risk for a given model w
	 * @param l
	 * @param w
	 * @return empirical risk
	 */
	protected double empiricalRisk(final List<STrainingSample<X, Y>> l, final double[] w) {
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
						STrainingSample<X, Y> ts = l.get(n);
						Y yp = prediction(ts.input);
						return delta(ts.output,yp);
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
			for(STrainingSample<X, Y> ts : l) {
				Y yp = prediction(ts.input);
				risk += delta(ts.output, yp);
			}
		}
		risk /= l.size();
		return risk;	
	}

	/**
	 * Compute the empirical risk
	 * @param l
	 * @return
	 */
	protected double empiricalRisk(List<STrainingSample<X, Y>> l) {
		return empiricalRisk(l,w);	
	}

	/**
	 * Compute &lt w, \psi(x,y) &gt 
	 * @param x
	 * @param y
	 * @return &lt w, \psi(x,y) &gt
	 */
	protected double valueOf(X x, Y y) {
		return valueOf(x,y,w);
	}

	/**
	 * Compute &lt w, \psi(x,y) &gt for a given w
	 * @param x
	 * @param y
	 * @param w
	 * @return &lt w, \psi(x,y) &gt
	 */
	protected double valueOf(X x, Y y, double[] w) {
		return linear.valueOf(w, psi(x,y));
	}

	/**
	 * Compute the loss augmented inference for a given trainign sample ts
	 * @param ts
	 * @return
	 */
	protected Y lossAugmentedInference(STrainingSample<X, Y> ts) {
		return lossAugmentedInference(ts, w);
	}

	/**
	 * Compute the prediction for a given input x
	 */
	@Override
	public Y prediction(X x) {
		return prediction(x,w);
	}

	/**
	 * Print the parameters
	 */
	protected void showParameters(){
		System.out.println("Train SSVM \tlambda: " + lambda + "\tdim: " + w.length + "\tthreads= " + nThreads);
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

