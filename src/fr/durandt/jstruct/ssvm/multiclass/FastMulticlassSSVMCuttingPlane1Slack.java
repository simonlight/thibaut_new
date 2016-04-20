/**
 * 
 */
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

import fr.durandt.jstruct.solver.MosekSolver;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;


/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class FastMulticlassSSVMCuttingPlane1Slack<X> extends FastMulticlassSSVM<X> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = -8900187396500347089L;

	/**
	 * Maximum number of cutting-plane models
	 */
	protected int cpmax = 500;
	/**
	 * Minimum number of cutting-plane models
	 */
	protected int cpmin = 5;

	/**
	 * Precision
	 */
	protected double epsilon = 1e-2;

	/**
	 * Time used by Mosek solver
	 */
	protected int solverTime = 0;

	/**
	 * Number of threads for Mosek solver
	 */
	protected int nThreadsMosek = 1;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Learning with cutting plane algorithm and 1 slack formulation
	 */
	@Override
	protected void learning(List<STrainingSample<X, Integer>> l) {

		double c = 1/lambda;
		int t=0;

		List<double[][]> lg = new ArrayList<double[][]>();
		List<Double> lc 	= new ArrayList<Double>();

		Object[] or 	= cuttingPlane(l,w);
		double[][] gt 	= (double[][]) or[0];
		double ct		= (Double) or[1];

		lg.add(gt);
		lc.add(ct);

		double[][] gram = null;
		double xi=0;

		while(t<cpmin || (t<=cpmax && VectorOp.dot(w,gt) < ct - xi - epsilon)) {

			System.out.print(".");
			if(t>0 && t % 100 == 0) {
				System.out.print(t);
			}
			if(t == cpmax) {
				System.out.print(" # max iter ");
			}

			if(gram != null) {
				double[][] g = gram;
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<g.length; i++) {
					for(int j=0; j<g.length; j++) {
						gram[i][j] = g[i][j];
					}
				}
				for(int i=0; i<lc.size(); i++) {
					gram[lc.size()-1][i] = VectorOp.dot(lg.get(lc.size()-1), lg.get(i));
					gram[i][lc.size()-1] = gram[lc.size()-1][i];
				}
				gram[lc.size()-1][lc.size()-1] += 1e-8;
			}
			else {
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = VectorOp.dot(lg.get(i), lg.get(j));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			long startTime = System.currentTimeMillis();
			double[] alphas = MosekSolver.solveQP(gram, lc, c, nThreadsMosek);
			long endTime = System.currentTimeMillis();
			solverTime += endTime - startTime;
			xi = (VectorOp.dot(alphas, lc.toArray(new Double[lc.size()])) - matrixProduct(alphas,gram))/c;

			// compute the new w
			w = new double[listClass.size()][dim];
			for(int i=0; i<alphas.length; i++) {
				for(int y=0; y<listClass.size(); y++) {
					for(int d=0; d<lg.get(i)[y].length; d++) {
						w[y][d] += alphas[i] * lg.get(i)[y][d];
					}
				}
			}
			t++;

			or = cuttingPlane(l, w);
			gt = (double[][]) or[0];
			ct = (Double) or[1];

			lg.add(gt);
			lc.add(ct);

		}
		System.out.println("*");
		if(verbose >= 0) {
			System.out.println("nb iterations= " + t + "\tloss aumented inference time= " + lossAugmentedTime/1000 + "s \t solver time= " + solverTime/1000 + "s");
		}
	}

	/**
	 * Compute the cutting plane model for a given w
	 * @param l
	 * @param w
	 * @return
	 */
	protected Object[] cuttingPlane(final List<STrainingSample<X, Integer>> l, final double[][] w) {

		// compute g(t) and c(t)
		double[][] gt = new double[listClass.size()][dim];
		double ct = 0;
		double n = l.size();

		long startTime = System.currentTimeMillis();

		if(nThreads > 1) {
			ExecutorService executor = Executors.newFixedThreadPool(nThreads);
			List<Future<Object[]>> futures = new ArrayList<Future<Object[]>>();
			CompletionService<Object[]> completionService = new ExecutorCompletionService<Object[]>(executor);

			for(int i=0 ; i<l.size(); i++) {
				final int ii = i;
				futures.add(completionService.submit(new Callable<Object[]>() {

					@Override
					public Object[] call() throws Exception {
						STrainingSample<X, Integer> ts = l.get(ii);
						Integer yp = lossAugmentedInference(ts, w);

						Object[] res = new Object[3];
						res[0] = yp;
						res[1] = delta(ts.output, yp);
						res[2] = ii;
						return res;
					}
				}));
			}

			for(Future<Object[]> f : futures) {
				try {
					Object[] res = f.get();
					Integer yp = (Integer)res[0];
					ct += (double)res[1];
					STrainingSample<X, Integer> ts = l.get((int) res[2]);
					double[] psiyp = psi(ts.input, yp);
					double[] psiyi = psi(ts.input, ts.output);
					for(int d=0; d<dim; d++) {
						gt[yp][d] 			+= -psiyp[d];
						gt[ts.output][d] 	+= psiyi[d];
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
				Integer yp = lossAugmentedInference(ts, w);
				ct += delta(ts.output, yp);

				double[] psiyp = psi(ts.input, yp);
				double[] psiyi = psi(ts.input, ts.output);
				for(int d=0; d<dim; d++) {
					gt[yp][d] 			+= -psiyp[d];
					gt[ts.output][d] 	+= psiyi[d];
				}
			}
		}
		ct /= n;

		for(int y=0; y<listClass.size(); y++) {
			for(int d=0; d<dim; d++) {
				gt[y][d] /= n;
			}
		}

		long endTime = System.currentTimeMillis();
		lossAugmentedTime += endTime - startTime;

		Object[] res = new Object[2];
		res[0] = gt;
		res[1] = ct;
		return res;
	}

	protected double matrixProduct(double[] alphas, double[][] gram) {
		// alpha^T*Gramm*alpha
		double[] tmp = new double[alphas.length];
		// tmp = gram * alpha
		for(int i=0; i<gram.length; i++) {
			tmp[i] = VectorOperations.dot(gram[i],alphas);
		}
		double s = VectorOperations.dot(alphas,tmp);
		return s;
	}

	@Override
	protected void showParameters() {
		super.showParameters();
		System.out.println("Learning: Cutting-Plane 1 Slack - Mosek");
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
	}

	@Override
	public String toString() {
		String s = "fast_multiclass_ssvm_cuttingplane1slack_lambda_" + lambda + "_epsilon_" + epsilon 
				+ "_cpmax_" + cpmax + "_cpmin_" + cpmin;
		return s;
	}

	/**
	 * @return the cpmax
	 */
	public int getCpmax() {
		return cpmax;
	}

	/**
	 * @param cpmax the cpmax to set
	 */
	public void setCpmax(int cpmax) {
		this.cpmax = cpmax;
	}

	/**
	 * @return the cpmin
	 */
	public int getCpmin() {
		return cpmin;
	}

	/**
	 * @param cpmin the cpmin to set
	 */
	public void setCpmin(int cpmin) {
		this.cpmin = cpmin;
	}

	/**
	 * @return the epsilon
	 */
	public double getEpsilon() {
		return epsilon;
	}

	/**
	 * @param epsilon the epsilon to set
	 */
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	/**
	 * @return the nThreadsMosek
	 */
	public int getnThreadsMosek() {
		return nThreadsMosek;
	}

	/**
	 * @param nThreadsMosek the nThreadsMosek to set
	 */
	public void setnThreadsMosek(int nThreadsMosek) {
		this.nThreadsMosek = nThreadsMosek;
	}



}
