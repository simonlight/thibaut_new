/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15;

import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class MantraPegasos<X,Y,H> extends Mantra<X,Y,H> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = -607649557560283938L;

	/**
	 * Maximum number of iterations
	 */
	protected int maxIter = 50;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Train SSVM with Pegasos
	 * @param l
	 */
	protected void learning(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		int s0 = 2*l.size();
		for(int iter=0; iter<maxIter; iter++) {
			boolean lastIteration = (iter+1)==maxIter;
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxIter + "\t");
			}
			else if(verbose == 0) {
				System.out.print(".");
				if(iter>0 && iter % 100 == 0) {
					System.out.print(iter);
				}
			}

			// Train 1 epoch
			trainOneEpoch(l, lastIteration, s0);

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
	public void trainOneEpoch(List<STrainingSample<LatentRepresentation<X,H>,Y>> l, boolean lastIteration, int s0) {

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

			// Compute the loss-augmented inference
			Object[] or = lossAugmentedInference(ts, w);
			Y yp = (Y)or[0];
			H hp = (H)or[1];
			H hm = (H)or[2];

			// Compute the "best" latent value for ground truth output
			Object[] res = prediction(ts.input.x, ts.output, w);
			H hpi = (H)res[0];
			H hmi = (H)res[1];

			// Compute the loss for sample s
			double loss = delta(ts.output, yp, hp) + valueOf(ts.input.x, yp, hp) + valueOf(ts.input.x, yp, hm) 
					- (valueOf(ts.input.x, ts.output, hpi) + valueOf(ts.input.x, ts.output, hmi));

			if(loss > 0) {
				updates++;
				double[] psi1 = psi(ts.input.x, yp, hp);
				double[] psi2 = psi(ts.input.x, yp, hm);
				double[] psi3 = psi(ts.input.x, ts.output, hpi);
				double[] psi4 = psi(ts.input.x, ts.output, hmi);
				eta /= l.size();
				for(int d=0; d<w.length; d++) {
					w[d] += (psi3[d] + psi4[d] - psi1[d] - psi2[d])*eta;
				}
			}

			// Projection step
			double proj = Math.min(1., Math.sqrt(2*l.size()/(lambda*Math.pow(VectorOp.getNormL2(w),2))));
			if(proj < 1) {
				projections++;
				for(int d=0; d<w.length; d++) {
					w[d] = w[d]*proj;
				}
			}

			// Take the average of all the generated solutions and use it as the final solution
			if(lastIteration) {
				for(int d=0; d<w.length; d++) {
					wbar[d] = (s*wbar[d] + w[d])/(s+1);
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
		System.out.println("Learning: Pegasos - max iterations= " + maxIter);
	}

	public String toString() {
		String s = "mantra_pegasos_lambda_" + lambda + "_maxIter_" + maxIter;
		return s;
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Getters and setters
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * @return the maxIter
	 */
	public int getMaxIter() {
		return maxIter;
	}

	/**
	 * @param maxIter the maxIter to set
	 */
	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}


}
