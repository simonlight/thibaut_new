/**
 * 
 */
package lsvmStandardVoc;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LSVMGradientDescent<X,H> extends LSVM<X,H> {

	protected int optim = 2	;
	

	protected int maxCCCPIter = 100;
	protected double epsilon = 1e-4;
	protected int minCCCPIter = 2;
	
	protected int maxEpochs = 100;
	
	protected boolean semiConvexity = true;
	protected boolean stochastic = false;
	
	protected double tradeoff;
	protected String gazeType;
	protected boolean hnorm;
	protected String className;
	
	protected HashMap<String , Double> lossMap = new HashMap<String , Double>(); 
	
	
	private long t=0;
	

	@Override
	protected void learn(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		if(optim == 1) {
			learnPegasos(l);
		}
		else if(optim == 2) {
			learnSGD(l);
		}
	}

	/**
	 * optim inspirée du ML3
	 * @param l
	 */
	protected void learnPegasos(List<TrainingSample<LatentRepresentation<X,H>>> l) {
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
			// update latent variables
			if(semiConvexity) {
				optimizePositiveLatent(l);
			}
			else {
				optimizeLatent(l);
			}
		}
		if(verbose == 0) {
			System.out.println("*");
		}

	}

	/**
	 * one epoch of SGD with Pegasos
	 * @param l
	 */
	public void trainOnceEpochsPegasos(List<TrainingSample<LatentRepresentation<X,H>>> l, boolean lastIteration, int s0) {

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

			TrainingSample<LatentRepresentation<X,H>> ts = l.get(s);
			if(semiConvexity && ts.label == -1) {
				ts.sample.h = optimizeH(ts.sample.x);
			}

			// Compute the loss for sample s
			double loss = loss(ts);

			// Compute the gradient
			if(loss > 0) {
				updates++;
				double[] psi = psi(ts.sample.x, ts.sample.h);
				eta *= ts.label;
				for(int i=0; i<w.length; i++) {
					w[i] += eta * psi[i];
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
			System.out.println("updates= " + updates + "\tprojections= " + projections + "\tobj= " + getPrimalObjective(l));
		}

		if(lastIteration) {
			w = wbar;
		}
	}

	/**
	 * 
	 * @param l
	 */
	
	
	protected void learnSGD(List<TrainingSample<LatentRepresentation<X,H>>> l) {

		// Shift t in order to have a
		// reasonable initial learning rate.
		// This assumes |x| \approx 1.
		double maxw = 1.0 / Math.sqrt(lambda);
		double typw = Math.sqrt(maxw);
		double eta0 = typw;
		t = (long) (1 / (eta0 * lambda));
		double oldPrimal_Objectif = 0;
		int iter = 0;
		double[] lastW = new double[dim];
		
		do {
			iter +=1;
			oldPrimal_Objectif = getPrimalObjective(l);
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else {
				System.out.print(".");
			}
			System.arraycopy(w, 0, lastW, 0, dim);
			for(int e=0; e<maxEpochs; e++) {
				trainOnceEpochsSGD(l);
			}
			// update latent variables
			if(semiConvexity) {
				optimizePositiveLatent(l);
			}
			else {
				optimizeLatent(l);
			}
		}while((oldPrimal_Objectif - getPrimalObjective(l)> epsilon || iter < minCCCPIter) && (iter < maxCCCPIter));
		System.out.format("total iteration %d times",iter);
		System.arraycopy(lastW, 0, w, 0, dim);
		
		if(verbose == 0) {
			System.out.println("*");
		}
	}

	
	/**
	 * Update the separating hyperplane by learning one epoch on given training list
	 * @param l the training list
	 */
	public void trainOnceEpochsSGD(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		
		if(w == null)
			return;
		
		int imax = l.size();
		if(stochastic) {
			Collections.shuffle(l);
		}
		
		for(int i=0; i<imax; i++) {
			
			double eta = 1.0 / (lambda * t);// learning rate
			
			double s = 1 - eta * lambda; // shrink
			//shrink w
			for(int d=0; d<w.length; d++) {
				w[d] *= s;
			}
			
			TrainingSample<LatentRepresentation<X,H>> ts = l.get(i);
			if(semiConvexity && ts.label == -1) {
				ts.sample.h = optimizeH(ts.sample.x);
			}

			//important: not the same as felzenswalb, note the objectif is without the number of examples n
			double[] x = psi(ts.sample.x, ts.sample.h);
			double y = ts.label;
			double z = y * linear.valueOf(w, x);

			if (z < 1) {
				eta *= y;
				for(int d=0; d<w.length; d++) {
					w[d] += x[d] * eta;
				}
			}
			t += 1;
		}
	}


	@Override
	protected void showParameters() {
		System.out.println("maxCCCPIter= " + maxCCCPIter + "\t maxEpochs= " + maxEpochs +
				"\t semi-convexity= " + semiConvexity + "\tstochastic= " + stochastic +
				"\t\t" + maxEpochs*maxCCCPIter);
		if(optim == 1) {
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
	 * @return the semiConvexity
	 */
	public boolean isSemiConvexity() {
		return semiConvexity;
	}

	/**
	 * @param semiConvexity the semiConvexity to set
	 */
	public void setSemiConvexity(boolean semiConvexity) {
		this.semiConvexity = semiConvexity;
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

	public double getEpsilon() {
		return epsilon;
	}

	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	public int getMinCCCPIter() {
		return minCCCPIter;
	}

	public void setMinCCCPIter(int minCCCPIter) {
		this.minCCCPIter = minCCCPIter;
	}
	
	public void setHnorm(boolean hnorm) {
		this.hnorm = hnorm;
	}
	public boolean getHnorm() {
		return hnorm;
	}

	public void setTradeOff(double tradeoff){
		this.tradeoff = tradeoff;
	}
	public double getTradeOff(){
		return tradeoff;
	}
	
	public void setCurrentClass(String className){
		this.className = className;
	}
	public String getCurrentClass(){
		return className;
	}
	
	
	public void setLossDict(String lossPath){
		
		try {
			ObjectInputStream is;
			is = new ObjectInputStream(new FileInputStream(lossPath));
			this.lossMap = (HashMap<String, Double> ) is.readObject();// 从流中读取User的数据  
			is.close();}
		catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  
		
	}

}
