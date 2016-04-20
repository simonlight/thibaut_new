/**
 * 
 */
package fr.durandt.jstruct.ssvm.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.solver.MosekSolver;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class ML3CuttingPlane1Slack extends ML3 {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8947859103307850959L;
	/**
	 * Maximum number of cutting-plane models
	 */
	protected int cpmax = 500;
	/**
	 * Minimum number of cutting-plane models
	 */
	protected int cpmin = 5;
	protected double epsilon = 1e-2;

	/**
	 * Learning with cutting plane algorithm and 1 slack formulation
	 */
	@Override
	protected void learning(List<STrainingSample<double[], Integer>> l) {

		double c = 1/lambda;
		int t=0;

		List<double[][][]> lg 	= new ArrayList<double[][][]>();
		List<Double> lc 		= new ArrayList<Double>();

		// Compute the initial cutting plane
		Object[] or 	= cuttingPlane(l,w);
		double[][][] gt = (double[][][]) or[0];
		double ct		= (Double) or[1];

		lg.add(gt);
		lc.add(ct);

		double[][] gram = null;
		double xi=0;

		while(t<cpmin || (t<=cpmax && dot(w,gt) < ct - xi - epsilon)) {

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
					gram[lc.size()-1][i] = dot(lg.get(lc.size()-1), lg.get(i));
					gram[i][lc.size()-1] = gram[lc.size()-1][i];
				}
				gram[lc.size()-1][lc.size()-1] += 1e-8;
			}
			else {
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = dot(lg.get(i), lg.get(j));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			double[] alphas = MosekSolver.solveQP(gram, lc, c);
			xi = (VectorOp.dot(alphas, lc.toArray(new Double[lc.size()])) - matrixProduct(alphas,gram))/c;
			System.out.println("DualObj= " + (VectorOp.dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram)) + "\talphas " + Arrays.toString(alphas));

			// new w
			w = new double[w.length][w[0].length][w[0][0].length];
			for(int i=0; i<alphas.length; i++) {
				for(int y : listClass) {
					for(int j=0; j<w[0].length; j++) {
						for(int k=0; k<w[0][0].length; k++) {
							w[y][j][k] += alphas[i] * lg.get(i)[y][j][k];
						}
					}
				}
			}
			t++;

			or = cuttingPlane(l, w);
			gt = (double[][][]) or[0];
			ct = (Double) or[1];

			lg.add(gt);
			lc.add(ct);

		}
		System.out.println("*");
	}

	/**
	 * Compute the cutting plane model for a given w
	 * @param l
	 * @param w
	 * @return
	 */
	protected Object[] cuttingPlane(List<STrainingSample<double[], Integer>> l, double[][][] w) {

		// compute g(t) and c(t)
		double[][][] gt = new double[c][m][d];
		double ct = 0;
		double n = l.size();

		for(STrainingSample<double[], Integer> ts : l){
			// Compute the optimal beta for yi
			//double[] betaStaryi = computeOptimalBeta(wt, ts.input, ts.output, p);
			double[] betaStaryi = computeOptimalBeta(w, ts.input, ts.output, p);

			Integer ybar = lossAugmentedInference(ts);
			// Compute the optimal beta for ybar
			double[] betaStarybar = computeOptimalBeta(w, ts.input, ybar, p);
			ct += delta(ts.output, ybar);;

			double[][] gradyi = computeMatrix(betaStaryi, ts.input);
			int y = ts.output;
			for(int j=0; j<w[y].length; j++) {
				for(int k=0; k<w[y][j].length; k++) {
					gt[y][j][k] += gradyi[j][k];
				}
			}

			double[][] gradybar = computeMatrix(betaStarybar, ts.input);
			y = ybar;
			for(int j=0; j<w[y].length; j++) {
				for(int k=0; k<w[y][j].length; k++) {
					gt[y][j][k] -= gradybar[j][k];
				}
			}
		}
		ct /= n;

		for(int y : listClass) {
			for(int j=0; j<w[y].length; j++) {
				for(int k=0; k<w[y][j].length; k++) {
					gt[y][j][k] /= n;
				}
			}
		}

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

	protected double dot(double[][][] a, double[][][] b) {
		double ps = 0;
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				for(int k=0; k<a[i][j].length; k++) {
					ps += a[i][j][k]*b[i][j][k];
				}
			}
		}
		return ps;
	}

	protected double delta(Integer yi, Integer y) {
		if(y == yi) {
			return 0;
		}
		else {
			return 1;
		}
	}

	public Integer lossAugmentedInference(STrainingSample<double[], Integer> ts) {
		int ypredict = -1;
		double max = -Double.MAX_VALUE;
		for(int y : listClass) {
			double val = valueOf(ts.input, y) + delta(ts.output, y);
			if(val > max) {
				max = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	@Override
	protected void showParametersLearning() {
		System.out.println("Learning: Cutting-Plane 1 Slack - Mosek");
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
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

	@Override
	public String toString() {
		return "ml3_cuttingplane1slack_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin + "_m_" + m + "_p_" + p;
	}

}
