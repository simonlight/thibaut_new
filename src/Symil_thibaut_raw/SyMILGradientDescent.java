package Symil_thibaut_raw;
/**
 * 
 */


import java.util.Collections;
import java.util.List;
import java.util.Random;

import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class SyMILGradientDescent<X,H> extends SyMIL<X,H> {

	protected int optim = 1;
	protected int maxCCCPIter = 50;
	protected int maxEpochs = 1;
	protected boolean stochastic = false;
	protected long t = 1;

	@Override
	protected void learn(List<TrainingSample<LatentRepresentation<X,H[]>>> l) {
		if(optim == 1) {
			learnSGD(l);
		}
		else if(optim == 2) {
			learnPegasos(l);
		}
	}

	/**
	 * optim inspirée du ML3
	 * @param l
	 */
	protected void learnPegasos(List<TrainingSample<LatentRepresentation<X,H[]>>> l) {

		int s0 = 2*l.size();
		for(int iter=0; iter<maxCCCPIter; iter++) {
			boolean lastIteration = (iter+1)==maxCCCPIter;

			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else {
				System.out.print(".");
			}

			for(int e=0; e<maxEpochs; e++) {
				trainOnceEpochsPegasos(l, lastIteration, s0);
			}
			s0 += 2*l.size();

			// update latent variables
			optimizeAllLatent(l);

		}
		if(verbose == 0) {
			System.out.println("*");
		}

	}

	/**
	 * one epoch of SGD with Pegasos
	 * @param l
	 */
	protected void trainOnceEpochsPegasos(List<TrainingSample<LatentRepresentation<X,H[]>>> l, boolean lastIteration, int s0) {

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

			TrainingSample<LatentRepresentation<X,H[]>> ts = l.get(s);
			// Compute latent variables
			H[] hpredict = optimizeH(ts.sample.x);
			H hp = hpredict[0];
			H hm = hpredict[1];

			// Compute gradients
			double[] xiPstar = psi(ts.sample.x, hp);
			double[] xiMstar = psi(ts.sample.x, hm);
			double[] xiP = psi(ts.sample.x, ts.sample.h[0]);
			double[] xiM = psi(ts.sample.x, ts.sample.h[1]);

			if(ts.label == 1) {
				double wx = linear.valueOf(w, xiPstar) - 1;
				if(wx > 0) {
					double e = eta/nbPlus;
					for (int d = 0; d < w.length; d++) {
						w[d] += - xiPstar[d] * e;
					}
				}

				double e = gamma * eta / nbPlus;
				if(1 - linear.valueOf(w, xiMstar) > linear.valueOf(w, xiPstar) ) {
					for (int d = 0; d < w.length; d++) {
						w[d] += e * xiMstar[d];
					}
				}
				else {
					for (int d = 0; d < w.length; d++) {
						w[d] += - e * xiPstar[d];
					}
				}

				e = eta * (1/(double)nbPlus + gamma/l.size());
				for (int d = 0; d < w.length; d++) {
					w[d] += e * xiP[d];
				}
			}
			else {
				double wx = -linear.valueOf(w, xiMstar) - 1;
				if(wx > 0) {
					double e = eta/nbMinus;
					for (int d = 0; d < w.length; d++) {
						w[d] += e * xiMstar[d];
					}
				}

				double e = gamma * eta / nbMinus;
				if(1 + linear.valueOf(w, xiPstar) > - linear.valueOf(w, xiMstar)) {
					for (int d = 0; d < w.length; d++) {
						w[d] += - e * xiPstar[d];
					}
				}
				else {
					for (int d = 0; d < w.length; d++) {
						w[d] += e * xiMstar[d];
					}
				}

				e = eta * (1/(double)nbMinus + gamma/l.size());
				for(int d=0; d<w.length; d++) {
					w[d] -= e * xiM[d];
				}
			}


			// Projection
			double proj = Math.min(1., Math.sqrt(2*l.size()/lambda) / VectorOperations.n2(w));
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
			System.out.println("updates= " + updates + "\tprojections= " + projections + "\tobj= " /*+ getPrimalObjective(l)*/);
		}

		if(lastIteration) {
			w = wbar;
		}
	}


	protected void learnSGD(List<TrainingSample<LatentRepresentation<X,H[]>>> l) {

		for(int el=0;el<maxCCCPIter;el++){

			//init
			// Shift t in order to have a
			// reasonable initial learning rate.
			// This assumes |x| \approx 1.
			double maxw = 1.0 / Math.sqrt(lambda);
			double typw = Math.sqrt(maxw);
			double eta0 = typw;
			t = (long) (1 / (eta0 * lambda));

			for(int e = 0 ; e < maxEpochs ; e++) {
				System.out.println("w^2:"+VectorOp.dot(w, w));
				trainOnceSGD(l);
			}
			System.out.println("*");
			optimizeAllLatent(l);
			//System.out.println(Arrays.toString(w));

		}
	}

	protected void trainOnceSGD(List<TrainingSample<LatentRepresentation<X,H[]>>> l) {

		if(w == null)
			return;

		int imax = l.size();
		if(stochastic) {
			Random seed = new Random(3);
			Collections.shuffle(l, seed);
//			Collections.shuffle(l);

		}

		for (int i = 0; i < imax; i++) {

			double eta = 1.0 / (lambda * t);
			double s = 1 - eta * lambda;
			for(int d=0; d<w.length; d++) {
				w[d] *= s;
			}

			TrainingSample<LatentRepresentation<X,H[]>> ts = l.get(i);
			// Compute latent variables
			H[] hpredict = optimizeH(ts.sample.x);
			H hp = hpredict[0];
			H hm = hpredict[1];

			// Compute gradients
			double[] xiPstar = psi(ts.sample.x, hp);
			double[] xiMstar = psi(ts.sample.x, hm);
			double vp = linear.valueOf(w, xiPstar);
			double vm = linear.valueOf(w, xiMstar);

			double[] xiP = psi(ts.sample.x, ts.sample.h[0]);
			double[] xiM = psi(ts.sample.x, ts.sample.h[1]);
			
			if(l.get(i).label == 1) {
				for(int d=0; d<w.length; d++) {
					w[d] += (1+gamma) * xiP[d] * eta;
				}
				
				double wx = vp - 1;
				
				if(wx > 0) {
					for(int d=0; d<w.length; d++) {
						w[d] += - xiPstar[d] * eta;
					}
				}

				if(1 - vm > vp) {
					for(int d=0; d<w.length; d++) {
						w[d] += gamma * xiMstar[d] * eta;
					}
				}
				else {
					for(int d=0; d<w.length; d++) {
						w[d] += - gamma * xiPstar[d] * eta ;
					}
				}

				
			}
			else {
				for(int d=0; d<w.length; d++) {
					w[d] += - (1+gamma) * xiM[d] * eta;
				}
				double wx = -vm - 1;
				if(wx > 0) {
					for(int d=0; d<w.length; d++) {
						w[d] += xiMstar[d] * eta;
					}
				}

				if(1 + vp > - vm) {
					for(int d=0; d<w.length; d++) {
						w[d] += - gamma * xiPstar[d] * eta;
					}
				}
				else {
					for(int d=0; d<w.length; d++) {
						w[d] += gamma * xiMstar[d] * eta;
					}
				}


			}

			t += 1;
		}

	}


	@Override
	protected void showParameters() {
		System.out.println("maxCCCPIter= " + maxCCCPIter + "\t maxEpochs= " + maxEpochs +
				"\tstochastic= " + stochastic +
				"\t\t" + maxEpochs*maxCCCPIter);
		if(optim == 1) {
			System.out.println("SGD");
		}
		else if(optim == 2) {
			System.out.println("Pegasos");
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

	/**
	 * @return the maxEpochs
	 */
	public int getMaxEpochs() {
		return maxEpochs;
	}

	/**
	 * @param maxEpochs the maxEpochs to set
	 */
	public void setMaxEpochs(int maxEpochs) {
		this.maxEpochs = maxEpochs;
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

	/**
	 * @return the stochastic
	 */
	public boolean isStochastic() {
		return stochastic;
	}

	/**
	 * @param stochastic the stochastic to set
	 */
	public void setStochastic(boolean stochastic) {
		this.stochastic = stochastic;
	}



}
