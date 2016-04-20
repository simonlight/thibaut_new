/**
 * 
 */
package fr.durandt.jstruct.ssvm.multiclass;

import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class ML3Pegasos extends ML3 {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6335622577837971320L;

	// number of CCCP iterations
	protected int maxCCCPIter = 30;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.multiclass.ML3#learn(java.util.List)
	 */
	@Override
	protected void learning(List<STrainingSample<double[], Integer>> l) {

		int s0 = 2*l.size();
		for(int iter=0; iter<maxCCCPIter; iter++) {

			boolean lastIteration = (iter+1)==maxCCCPIter;

			// Save the previous solution in order to be able to compute the first-order taylor approximation of the score
			double[][][] wt = w.clone();

			if(verbose > 0) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else {
				System.out.print(".");
			}

			// Train 1 epoch with Pegasos
			trainOneEpochs(l, lastIteration, s0, wt);
			s0 += 2*l.size();

		}
		if(verbose == 0) {
			System.out.println("*");
		}

	}

	/**
	 * one epoch of SGD with Pegasos
	 * @param l
	 */
	public void trainOneEpochs(List<STrainingSample<double[], Integer>> l, boolean doAvg, int s0, double[][][] wt) {

		double[][][] wbar = null;
		if(doAvg) {
			wbar = new double[c][m][d];
		}

		int updates = 0;
		int projections = 0;

		Collections.shuffle(l);
		double eta = 0;
		for(int s=0; s<l.size(); s++) {

			eta = 1./(lambda*(s+1+s0));


			// Compute the optimal beta for yi
			double[] betaStaryi = computeOptimalBeta(wt, l.get(s).input, l.get(s).output, p);

			// Compute the output with the highest score and different of yi
			// ybar = argmax_{y != yi} max_{beta} beta^T W_y x
			int ybar = prediction(l.get(s).input, l.get(s).output);
			// Compute the optimal beta for ybar
			double[] betaStarybar = computeOptimalBeta(w, l.get(s).input, ybar, p);

			double reg=(1-eta*lambda);
			for(int y : listClass) {
				for(int j=0; j<w[y].length; j++) {
					for(int k=0; k<w[y][j].length; k++) {
						w[y][j][k] *= reg;
					}
				}
			}

			// Compute the loss for sample s
			double loss = Math.max(0, 1 + valueOf(l.get(s).input, betaStarybar, ybar) 
					- valueOf(l.get(s).input, betaStaryi, l.get(s).output));

			if(loss > 0) {
				updates++;

				double[][] gradyi = computeMatrix(betaStaryi, l.get(s).input);
				int y = l.get(s).output;
				for(int j=0; j<w[y].length; j++) {
					for(int k=0; k<w[y][j].length; k++) {
						w[y][j][k] += gradyi[j][k]*eta;
					}
				}

				double[][] gradybar = computeMatrix(betaStarybar, l.get(s).input);
				y = ybar;
				for(int j=0; j<w[y].length; j++) {
					for(int k=0; k<w[y][j].length; k++) {
						w[y][j][k] -= gradybar[j][k]*eta;
					}
				}
			}

			// Projection
			double proj = Math.min(1., Math.sqrt(2*l.size()/lambda)/ frobenius(w));
			if(proj < 1) {
				projections++;
				for(int i=0; i<w.length; i++) {
					for(int j=0; j<w[i].length; j++) {
						for(int k=0; k<w[i][j].length; k++) {
							w[i][j][k] = w[i][j][k]*proj;
						}
					}
				}
			}

			// Take the average of all the generated solutions and use it as the final solution
			if(doAvg) {
				for(int i=0; i<w.length; i++) {
					for(int j=0; j<w[i].length; j++) {
						for(int k=0; k<w[i][j].length; k++) {
							wbar[i][j][k] = (s*wbar[i][j][k] + w[i][j][k])/(s+1);
						}
					}
				}
			}
		}

		if(verbose > 0) {
			System.out.println("updates= " + updates + "\tprojections= " + projections + "\tobj= " + primalObj(l));
			evaluation(l);
		}

		if(doAvg) {
			w = wbar;
		}
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

	@Override
	public String toString() {
		return "ml3_pegasos_lambda_" + lambda + "_maxCCCPIter_" + maxCCCPIter + "_m_" + m + "_p_" + p;
	}


}
