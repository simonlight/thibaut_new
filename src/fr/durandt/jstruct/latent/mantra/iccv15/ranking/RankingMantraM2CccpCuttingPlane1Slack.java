/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.solver.MosekSolver;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class RankingMantraM2CccpCuttingPlane1Slack<X,H> extends RankingMantraM2<X,H> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = -4949268968209924039L;

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
	
	/**
	 * Maximum number of CCCP iterations
	 */
	protected int maxCCCPIter = 50;

	/**
	 * Minimum number of CCCP iterations
	 */
	protected int minCCCPIter = 2;
	
	protected double gamma = 1.;

	/**
	 * 1/|P||N| * sum_{i in P} sum_{j in N} &lt w, phi(xi,hi^+) + phi(xi,hi^-) &gt - &lt w, phi(xj,hj^+) + phi(xj,hj^-) &gt
	 * @param x
	 * @param h
	 * @return
	 */
	protected abstract double[] psi(X x, RankingOutput y, H h);

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Learning with cutting plane algorithm and 1 slack formulation
	 */
	@Override
	protected void learning(List<STrainingSample<LatentRepresentation<X,H>,RankingOutput>> l) {
		int el=0;
		double decrement = 0;
		double precObj = 0;

		// linearize the concave part
		double[] wt = w.clone(); 

		while(el<minCCCPIter || (el<=maxCCCPIter && decrement < 0)) {
			System.out.println("epoch latent " + el);

			// solve the convexified optimization problem 
			optimizeConvexifiedProblem(l.get(0), wt);

			double obj = primalObj(l);
			decrement = obj - precObj;
			System.out.println("obj= " + obj + "\tdecrement= " + decrement);
			precObj = obj;
			el++;

			// linearize the concave part
			wt = w.clone(); 
		}
	}

	/**
	 * Learning with convex cutting plane algorithm and 1 slack formulation
	 */
	protected void optimizeConvexifiedProblem(STrainingSample<LatentRepresentation<X,H>,RankingOutput> ts, double[] wt) {

		double c = lambda != 0 ? 1./lambda : 0.0;
		int t = 0;

		List<double[]> lg = new ArrayList<double[]>();
		List<Double> lc 	= new ArrayList<Double>();

		// Compute initial cutting plane
		Object[] or 	= cuttingPlane(ts, w);
		double[] gt 	= (double[]) or[0];
		double ct		= (Double) or[1];
		
		VectorOp.mul(wt, (2*gamma) / (c * Math.sqrt(2*gamma + 1)));

		VectorOp.mul(gt, 1./Math.sqrt(1+2*gamma));
		VectorOp.add(gt, wt);

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

			if(verbose > 2) {
				System.out.println("alphas= " + Arrays.toString(alphas));
				System.out.println("DualObj= " + (VectorOp.dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram)));
			}

			// new w
			// Initialize w = 0
			for(int d=0; d<dim; d++) {
				w[d] = 0.0;
			}
			// Compute the new w
			for(int i=0; i<alphas.length; i++) {
				for(int d=0; d<dim; d++) {
					w[d] += alphas[i] * lg.get(i)[d];
				}
			}
			t++;

			// Compute new cutting-plane
			or = cuttingPlane(ts, w);
			gt = (double[]) or[0];
			ct = (Double) or[1];
			
			VectorOp.mul(gt, 1./Math.sqrt(1+2*gamma));
			VectorOp.add(gt, wt);

			lg.add(gt);
			lc.add(ct);

		}
		System.out.println("*");
		if(verbose >= 0) {
			System.out.println("cutting-planes= " + t + "\tloss aumented inference time= " + lossAugmentedTime/1000 + "s \tsolver time= " + solverTime/1000 + "s");
		}
	}

	/**
	 * Compute the cutting plane for the current model w
	 * @param l list of training samples
	 * @return the cutting plane model
	 */
	public Object[] cuttingPlane(final STrainingSample<LatentRepresentation<X,H>,RankingOutput> ts, final double[] w) {
		// compute g(t) and c(t)
		final double[] gt = new double[dim];
		double ct = 0;

		long startTime = System.currentTimeMillis();

		// Compute the loss-augmented inference
		Object[] or = lossAugmentedInference(ts, w);
		RankingOutput yp = (RankingOutput)or[0];
		H hp = (H)or[1];

		// Linear term of the cutting plane model
		ct += delta(ts.output, yp, hp);

		// Vector term of the cutting plane model
		double[] psi1 = psi(ts.input.x, yp, hp);

		// 
		H hi = prediction(ts.input.x, ts.output); // hi == hp ?
		double[] psi2 = psi(ts.input.x, ts.output, hi);

		//double score = delta(ts.output, yp, hp) + linear.valueOf(w, psi1) - linear.valueOf(w, psi2);

		for(int d=0; d<dim; d++) {
			gt[d] += psi2[d] - psi1[d];
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
		System.out.println("Optimization: CCCP + Cutting-Plane 1 Slack with Mosek solver");
		System.out.println("epsilon= " + epsilon 
				+ "\tmaxCCCPIter= " + maxCCCPIter + "\tminCCCPIter= " + minCCCPIter 
				+ "\tcpmax= " + cpmax + "\tcpmin= " + cpmin + "\tgamma= " + gamma);
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {

		String s = "ranking_mantra_cccp_cuttingplane1slack_lambda_" + lambda 
				+ "_maxCCCPIter_" + maxCCCPIter + "_minCCCPIter_" + minCCCPIter
				+ "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin + "_gamma_" + gamma;;
		return s;
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

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

	/**
	 * @return the maxCCCPIter
	 */
	public int getMaxCCCPIter() {
		return maxCCCPIter;
	}

	/**
	 * @param maxCCCPIter the maxCCCPIter to set
	 */
	public void setMaxCCCPIter(int maxCCCPIter) {
		this.maxCCCPIter = maxCCCPIter;
	}

	/**
	 * @return the minCCCPIter
	 */
	public int getMinCCCPIter() {
		return minCCCPIter;
	}

	/**
	 * @param minCCCPIter the minCCCPIter to set
	 */
	public void setMinCCCPIter(int minCCCPIter) {
		this.minCCCPIter = minCCCPIter;
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
