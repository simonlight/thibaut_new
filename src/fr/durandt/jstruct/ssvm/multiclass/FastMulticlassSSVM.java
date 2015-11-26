package fr.durandt.jstruct.ssvm.multiclass;

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
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;

public abstract class FastMulticlassSSVM<X> implements StructuralClassifier<X, Integer> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = 2037200351810569270L;

	protected double lambda = 1e-4;

	protected int verbose = 0;

	/**
	 * Time used to compute the loss augmented inference
	 */
	protected int lossAugmentedTime = 0;

	/**
	 * svm hyperplane
	 */
	protected double[][] w = null;

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

	/**
	 * List of classes {0,...,c-1} where c is the number of classes
	 */
	protected List<Integer> listClass = null;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Abstract methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	protected abstract void learning(List<STrainingSample<X, Integer>> l);
	protected abstract Integer lossAugmentedInference(STrainingSample<X, Integer> ts, double[][] w);
	protected abstract Integer prediction(X x, double[][] w);
	protected abstract double delta(Integer yi, Integer y);
	protected abstract double[] psi(X x, Integer y);

	/**
	 * This method must: <br />
	 * - define the dimension of w <br />
	 * - initialize w
	 * @param l
	 */
	protected abstract void init(List<STrainingSample<X, Integer>> l);


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	@Override
	public void train(List<STrainingSample<X, Integer>> l) {

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
		// Compute empirical risk
		double risk = empiricalRisk(l);
		System.out.println("Empirical risk= " + risk);
		// Compute the multiclass accuracy
		double acc = multiclassAccuracy(l);
		System.out.println("Multiclass accuracy= " + acc);
		System.out.println("----------------------------------------------------------------------------------------");
	}

	/**
	 * Compute the loss function for a given w
	 * @param l
	 * @param w
	 * @return the loss function
	 */
	protected double loss(final List<STrainingSample<X, Integer>> l, final double[][] w) {
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
						STrainingSample<X, Integer> ts = l.get(n);
						Integer yp = lossAugmentedInference(ts, w);
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
			for(STrainingSample<X, Integer> ts : l) {
				Integer yp = lossAugmentedInference(ts, w);
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
	protected double loss(List<STrainingSample<X, Integer>> l) {
		return loss(l, w);
	}

	/**
	 * Compute the primal objective for given w
	 * @param l
	 * @param w
	 * @return
	 */
	protected double primalObj(List<STrainingSample<X, Integer>> l, double[][] w) {
		double obj = lambda * VectorOp.dot(w,w)/2;
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
	public double primalObj(List<STrainingSample<X, Integer>> l) {
		return primalObj(l, w);
	}

	/**
	 * Compute the empirical risk
	 * @param l
	 * @return empirical risk
	 */
	protected double empiricalRisk(final List<STrainingSample<X, Integer>> l) {
		double risk = 0;
		if(nThreads > 1) {
			ExecutorService executor = Executors.newFixedThreadPool(nThreads);
			List<Future<Double>> futures = new ArrayList<Future<Double>>();
			CompletionService<Double> completionService = new ExecutorCompletionService<Double>(executor);

			for(int i=0 ; i<l.size(); i++) {
				final int n = i;
				futures.add(completionService.submit(new Callable<Double>() {

					@Override
					public Double call() throws Exception {
						STrainingSample<X, Integer> ts = l.get(n);
						Integer yp = prediction(ts.input);
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
		else {
			for(STrainingSample<X, Integer> ts : l) {
				Integer yp = prediction(ts.input);
				risk += delta(ts.output,yp);
			}
		}
		risk /= l.size();
		return risk;	
	}

	/**
	 * Compute &lt w, \psi(x,y) &gt 
	 * @param x
	 * @param y
	 * @return &lt w, \psi(x,y) &gt
	 */
	protected double valueOf(X x, Integer y) {
		return valueOf(x, y, w);
	}

	/**
	 * Compute &lt w, \psi(x,y) &gt for a given w
	 * @param x
	 * @param y
	 * @param w
	 * @return &lt w, \psi(x,y) &gt
	 */
	protected double valueOf(X x, Integer y, double[][] w) {
		return linear.valueOf(w[y], psi(x, y));
	}

	/**
	 * Compute the loss augmented inference
	 * @param ts
	 * @return
	 */
	protected Integer lossAugmentedInference(STrainingSample<X, Integer> ts) {
		return lossAugmentedInference(ts, w);
	}

	public Integer prediction(X x) {
		return prediction(x,w);
	}

	/**
	 * Print the paramters
	 */
	protected void showParameters(){
		System.out.println("Train Multiclass SSVM \tlambda: " + lambda + "\tw: " + listClass.size() + "x" + dim + "\tthreads= " + nThreads);
	}

	public double multiclassAccuracy(final List<STrainingSample<X, Integer>> l) {
		double accuracy = 0;
		int nb = 0;
		if(nThreads > 1) {
			ExecutorService executor = Executors.newFixedThreadPool(nThreads);
			List<Future<Integer>> futures = new ArrayList<Future<Integer>>();
			CompletionService<Integer> completionService = new ExecutorCompletionService<Integer>(executor);

			for(int i=0 ; i<l.size(); i++) {
				final int n = i;
				futures.add(completionService.submit(new Callable<Integer>() {

					@Override
					public Integer call() throws Exception {
						STrainingSample<X, Integer> ts = l.get(n);
						int ypredict = prediction(ts.input);
						if(ts.output == ypredict){	
							return 1;
						}
						return 0;
					}
				}));
			}

			for(Future<Integer> f : futures) {
				try {
					Integer res = f.get();
					if (res != null) {
						nb += res;
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			executor.shutdown();
		}
		else {
			for(STrainingSample<X, Integer> ts : l){
				int ypredict = prediction(ts.input);
				if(ts.output == ypredict){	
					nb++;
				}
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * @return the verbose
	 */
	public int getVerbose() {
		return verbose;
	}

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
	public double[][] getW() {
		return w;
	}

	/**
	 * @param w the w to set
	 */
	public void setW(double[][] w) {
		this.w = w;
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
		this.nThreads = nThreads;
	}

	/**
	 * @param verbose the verbose to set
	 */
	public void setVerbose(int verbose) {
		this.verbose = verbose;
	}

}
