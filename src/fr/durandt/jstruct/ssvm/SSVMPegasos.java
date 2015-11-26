/**
 * 
 */
package fr.durandt.jstruct.ssvm;

import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;

/**
 * <b>Pegasos: Primal Estimated sub-GrAdient SOlver for SVM</b><br/>
 * Shai S. Shwartz, Yoram Singer, Nathan Srebro<br/>
 * <i>International Conference on Machine learning (2007)</i>
 * </p>
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class SSVMPegasos<X,Y> extends SSVM<X,Y> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = 1774471833614026694L;

	/**
	 * Maximum number of iterations
	 */
	protected int maxIterations = 50;
	
	/**
	 * 
	 */
	protected boolean stochastic = true;

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Train SSVM with Pegasos
	 * @param l
	 */
	protected void learning(List<STrainingSample<X, Y>> l) {
		int s0 = 2*l.size();
		for(int iter=0; iter<maxIterations; iter++) {
			boolean lastIteration = (iter+1)==maxIterations;
			
			// print the number of iterations
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxIterations + "\t");
			}
			else if(verbose == 0) {
				System.out.print(".");
				if(iter>0 && iter % 100 == 0) {
					System.out.print(iter);
				}
			}
			
			// train 1 epoch
			trainOneEpoch(l, lastIteration, s0);
			
			// update learning rate
			s0 += 2*l.size();
		}
		if(verbose == 0) {
			System.out.println("*");
		}
	}

	/**
	 * Train one epoch with Pegasos
	 * @param l
	 */
	public void trainOneEpoch(List<STrainingSample<X, Y>> l, boolean lastIteration, int s0) {

		double[] wbar = null;
		if(lastIteration) {
			wbar = new double[w.length];
		}

		int updates = 0;
		int projections = 0;

		if(stochastic) {
			Collections.shuffle(l);
		}
		double eta = 0;
		for(int s=0; s<l.size(); s++) {
			eta = 1./(lambda*(s+1+s0));

			double g = 1 - eta*lambda;
			for(int i=0; i<w.length; i++) {
				w[i] = w[i]*g;
			}

			STrainingSample<X, Y> ts = l.get(s);
			Y yp = lossAugmentedInference(ts, w);

			// Compute the loss for sample s
			double loss = delta(ts.output, yp) + valueOf(ts.input, yp) - valueOf(ts.input, ts.output);

			if(loss > 0) {
				updates++;
				double[] psi1 = psi(ts.input, yp);
				double[] psi2 = psi(ts.input, ts.output);
				eta /= l.size();
				for(int i=0; i<w.length; i++) {
					w[i] += (psi2[i] - psi1[i])*eta;
				}
			}

			// Projection step
			double proj = Math.min(1., Math.sqrt(2*l.size()/(lambda*Math.pow(VectorOp.getNormL2(w),2))));
			if(proj < 1) {
				projections++;
				for(int i=0; i<w.length; i++) {
					w[i] = w[i]*proj;
				}
			}

			// Take the average of all the generated solutions and use it as the final solution
			if(lastIteration) {
				for(int i=0; i<w.length; i++) {
					wbar[i] = (s*wbar[i] + w[i])/(s+1);
				}
			}
		}

		if(verbose >= 1) {
			System.out.println("updates= " + updates + "\tprojections= " + projections + "\tobj= " + primalObj(l));
		}

		if(lastIteration) {
			w = wbar;
		}
	}

	protected void showParameters() {
		super.showParameters();
		System.out.println("Learning: Pegasos - max iterations= " + maxIterations);
	}

	public String toString() {
		String s = "ssvm_pegasos_lambda_" + lambda + "_maxIter_" + maxIterations;
		return s;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * @return the maxIterations
	 */
	public int getMaxIterations() {
		return maxIterations;
	}

	/**
	 * @param maxIterations the maxIterations to set
	 */
	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	/**
	 * @return the stochastic
	 */
	public boolean isStochastic() {
		return stochastic;
	}

	/**
	 * @param stochastic the stochastic to set (true = stochastic)
	 */
	public void setStochastic(boolean stochastic) {
		this.stochastic = stochastic;
	}


}
