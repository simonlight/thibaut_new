/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm;

import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LSSVMPegasos<X,Y,H> extends LSSVM<X,Y,H> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = -7691619022390377640L;

	/**
	 * Maximum number of CCCP iterations
	 */
	protected int maxCCCPIter = 50;

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Train SSVM with Pegasos
	 * @param l
	 */
	@Override
	protected void learning(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		int s0 = 2*l.size();
		for(int iter=0; iter<maxCCCPIter; iter++) {
			boolean lastIteration = (iter+1)==maxCCCPIter;
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else if(verbose == 0) {
				System.out.print(".");
				if(iter>0 && iter % 100 == 0) {
					System.out.print(iter);
				}
			}

			// Save the previous solution in order to be able to compute the 
			// first-order Taylor approximation of the score
			double[] wt = w.clone();

			// Train 1 epoch
			trainOneEpoch(l, lastIteration, s0, wt);

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
	public void trainOneEpoch(List<STrainingSample<LatentRepresentation<X,H>,Y>> l, boolean lastIteration, int s0, double[] wt) {

		double[] wbar = null;
		if(lastIteration) {
			wbar = new double[w.length];
		}

		int updates = 0;
		int projections = 0;

		Collections.shuffle(l);
		double eta = 0;
		for(int s=0; s<l.size(); s++) {
			eta = 1./(lambda*(s+1+s0));

			double g = 1 - eta*lambda;
			for(int i=0; i<w.length; i++) {
				w[i] = w[i]*g;
			}

			STrainingSample<LatentRepresentation<X,H>,Y> ts = l.get(s);

			// Compute the loss augmented inference
			Object[] or = lossAugmentedInference(ts);
			Y yp = (Y)or[0];
			H hp = (H)or[1];
			// Compute the "best" latent value for ground truth output
			ts.input.h = prediction(ts.input.x, ts.output, wt);

			// Compute the loss for sample s
			double loss = delta(ts.output, yp, hp) + valueOf(ts.input.x, yp, hp) 
					- valueOf(ts.input.x, ts.output, ts.input.h);

			if(loss > 0) {
				updates++;
				double[] psi1 = psi(ts.input.x, yp, hp);
				double[] psi2 = psi(ts.input.x, ts.output, ts.input.h);
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
	
	@Override
	protected void showParameters() {
		super.showParameters();
		System.out.println("Learning: Pegasos - max CCCP iterations= " + maxCCCPIter);
	}

	@Override
	public String toString() {
		String s = "lssvm_pegasos_lambda_" + lambda + "_maxCCCPIter_" + maxCCCPIter;
		return s;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////


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
