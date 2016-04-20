/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.multiclass;

import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LatentML3Pegasos<X,H> extends LatentML3<X,H> {

	// number of CCCP iterations
	protected int maxCCCPIter = 30;

	/**
	 * 
	 */
	private static final long serialVersionUID = 3733838105784158135L;

	@Override
	protected void learn(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		int s0 = 2*l.size();
		for(int iter=0; iter<maxCCCPIter; iter++) {

			boolean lastIteration = (iter+1)==maxCCCPIter;

			// Save the previous solution in order to be able to compute the first-order taylor approximation of the score
			double[][][] wt = w.clone();

			if(verbose > 0) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}

			// Train 1 epoch with Pegasos
			trainOneEpochs(l, lastIteration, s0, wt);
			s0 += 2*l.size();

		}
	}

	/**
	 * one epoch of SGD with Pegasos
	 * @param l
	 */
	public void trainOneEpochs(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l, boolean doAvg, int s0, double[][][] wt) {

		double[][][] wbar = null;
		if(doAvg) {
			wbar = new double[c][m][d];
		}

		int updates = 0;
		int projections = 0;

		Collections.shuffle(l);
		double eta = 0;
		for(int s=0; s<l.size(); s++) {

			STrainingSample<LatentRepresentation<X,H>,Integer> ts = l.get(s);

			eta = 1./(lambda*(s+1+s0));

			// Compute the optimal beta for yi
			H hStaryi = prediction(ts.input.x, ts.output, wt);
			double[] betaStaryi = computeOptimalBeta(wt, ts.input.x, ts.output, hStaryi, p);

			// Compute the output with the highest score and different of yi
			// ybar = argmax_{y != yi} max_{beta} beta^T W_y x
			int ybar = predictionLAI(ts);
			H hStarybar = prediction(ts.input.x, ybar);
			// Compute the optimal beta for ybar
			double[] betaStarybar = computeOptimalBeta(w, ts.input.x, ybar, hStarybar, p);

			double reg=(1-eta*lambda);
			for(int y : listClass) {
				for(int j=0; j<w[y].length; j++) {
					for(int k=0; k<w[y][j].length; k++) {
						w[y][j][k] *= reg;
					}
				}
			}

			// Compute the loss for sample s
			double loss = Math.max(0, 1 + valueOf(ts.input.x, ybar, hStarybar, betaStarybar) 
					- valueOf(ts.input.x, ts.output, hStaryi, betaStaryi));

			if(loss > 0) {
				updates++;

				double[][] gradyi = computeMatrix(betaStaryi, psi(ts.input.x, hStaryi));
				int y = ts.output;
				for(int j=0; j<w[y].length; j++) {
					for(int k=0; k<w[y][j].length; k++) {
						w[y][j][k] += gradyi[j][k]*eta;
					}
				}

				double[][] gradybar = computeMatrix(betaStarybar, psi(ts.input.x, hStarybar));
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

	@Override
	public String toString() {
		return "latent_ml3_pegasos_lambda_" + lambda + "_maxCCCPIter_" + maxCCCPIter + "_m_" + m + "_p_" + p;
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



}
