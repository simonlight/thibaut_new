/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.solver.MosekSolver;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class MantraCuttingPlane1Slack<X,Y,H> extends Mantra<X,Y,H> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = 452830233892631602L;

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
	 * Time used to pre-compute scores
	 */
	protected int precomputeTime = 0;

	/**
	 * Number of threads for Mosek solver
	 */
	protected int nThreadsMosek = 1;

	/**
	 * Optimization type: <br/>
	 * 1 non-convex 1 slack cutting-plane <br/>
	 * 2 convex 1 slack cutting-plane (the optimization problem is non-convex) <br/>
	 */
	protected int optim = 1;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Learning with cutting plane algorithm and 1 slack formulation
	 */
	@Override
	protected void learning(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		if(optim == 1) {
			trainIterNonConvex1SlackCuttingPlane(l);
		}
		else if(optim == 2) {
			trainIterConvex1SlackCuttingPlane(l);
		}
		else {
			System.out.println("ERROR Optim option invalid " + optim);
			System.exit(0);
		}
	}

	protected void trainIterNonConvex1SlackCuttingPlane(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		System.out.println("todo");
		System.exit(0);
	}

	/**
	 * Learning with convex cutting plane algorithm and 1 slack formulation
	 */
	protected void trainIterConvex1SlackCuttingPlane(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {

		double c = lambda != 0 ? 1./lambda : 0.0;
		int t = 0;

		List<double[]> lg = new ArrayList<double[]>();
		List<Double> lc 	= new ArrayList<Double>();

		// Compute initial cutting plane
		Object[] or 	= cuttingPlane(l, w);
		double[] gt 	= (double[]) or[0];
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
			or = cuttingPlane(l, w);
			gt = (double[]) or[0];
			ct = (Double) or[1];

			lg.add(gt);
			lc.add(ct);

		}
		System.out.println("*");
		if(verbose >= 0) {
			System.out.println("cutting-planes= " + t + "\tloss aumented inference time= " + lossAugmentedTime/1000 + "s \tpre-compute scores= " + precomputeTime/1000 + "s \tsolver time= " + solverTime/1000 + "s");
		}
	}

	/**
	 * Compute the cutting plane for the current model w
	 * @param l list of training samples
	 * @return the cutting plane model
	 */
	public Object[] cuttingPlane(final List<STrainingSample<LatentRepresentation<X,H>,Y>> l, final double[] w) {
		// compute g(t) and c(t)
		final double[] gt = new double[dim];
		double ct = 0;
		double n = l.size();

		long startTime = System.currentTimeMillis();

		for(int i=0; i<l.size(); i++){
			STrainingSample<LatentRepresentation<X,H>,Y> ts = l.get(i);	

			// Compute the loss-augmented inference
			Object[] or = lossAugmentedInference(ts, w);
			Y yp = (Y)or[0];
			H hp = (H)or[1];
			H hm = (H)or[2];

			// Linear term of the cutting plane model
			ct += delta(ts.output, yp, hp);

			// Vector term of the cutting plane model
			double[] psi1 = psi(ts.input.x, yp, hp);
			double[] psi2 = psi(ts.input.x, yp, hm);

			// Compute the "best" latent value for ground truth output
			Object[] res = prediction(ts.input.x, ts.output, w);
			H hpi = (H)res[0];
			H hmi = (H)res[1];

			double[] psi3 = psi(ts.input.x, ts.output, hpi);
			double[] psi4 = psi(ts.input.x, ts.output, hmi);

			for(int d=0; d<dim; d++) {
				gt[d] += -psi1[d] - psi2[d] + psi3[d] + psi4[d];
			}
		}
		ct /= n;

		for(int d=0; d<dim; d++) {
			gt[d] /= n;
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
			tmp[i] = VectorOperations.dot(gram[i], alphas);
		}
		double s = VectorOperations.dot(alphas, tmp);
		return s;
	}

	@Override
	protected void showParameters() {
		super.showParameters();
		if(optim == 1) {
			System.out.println("Learning: Non-Convex Cutting-Plane 1 Slack - Mosek");
		}
		else if(optim == 2) {
			System.out.println("Learning: Convex Cutting-Plane 1 Slack - Mosek");
		}
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		String s = "mantra_cuttingplane1slack_optim_" + optim + "_lambda_" + lambda 
				+ "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin;
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
	 * @return the optim
	 */
	public int getOptim() {
		return optim;
	}

	/**
	 * @param optim the optim to set
	 */
	public void setOptim(int optim) {
		this.optim = optim;
	}


}
