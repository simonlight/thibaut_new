/**
 * 
 */
package Symil_Musk;

import java.util.Collections;
import java.util.List;
import fr.durandt.jstruct.latent.LatentRepresentationSymil;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LSVMGradientDescent<X,H> extends LSVM<X,H> {

	protected int optim;
	

	protected int maxCCCPIter ;
	protected double epsilon;
	protected int minCCCPIter;
	protected double nbd;
	protected int maxEpochs;
	
	protected boolean stochastic;
	
	protected double tradeoff;
	protected String gazeType;
	protected boolean hnorm;
	protected String className;
	
	private long t=1;
	
	@Override
	protected void learn(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l) {
		if(optim == 2) {
			learnSGD(l);
		}
		if(optim == 1) {
			learnPegasos(l);
		}
	}

	protected void learnPegasos(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l) {

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
				optimizeLatent(l);

			}
			s0 += 2*l.size();

			// update latent variables
			optimizeLatent(l);

		}
		if(verbose == 0) {
			System.out.println("*");
		}

	}

	/**
	 * one epoch of SGD with Pegasos
	 * @param l
	 */
	protected void trainOnceEpochsPegasos(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l, boolean lastIteration, int s0) {

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
		double[][] positiveBatchPsi = new double[l.size()][dim];		
		double[][] negativeBatchPsi = new double[l.size()][dim];		
		for (int i=0; i<l.size();i++){
			double[] cp2Psi = psi(l.get(i).sample.x, l.get(i).sample.hp);
			double[] cn2Psi = psi(l.get(i).sample.x, l.get(i).sample.hn);
			positiveBatchPsi[i]=cp2Psi;
			negativeBatchPsi[i]=cn2Psi;
		}
		for(int s=0; s<l.size(); s++) {
			eta = 1./(lambda*(s+1+s0));

			double g = 1 - eta*lambda;
			for(int i=0; i<w.length; i++) {
				w[i] = w[i]*g;
			}

			TrainingSample<LatentRepresentationSymil<X, H, H>> ts = l.get(s);
			// Compute latent variables
			H[] hpredict = optimizeH(ts.sample.x);
			H hp = hpredict[0];
			H hm = hpredict[1];

			// Compute gradients
			double[] xiPstar = psi(ts.sample.x, hp);
			double[] xiMstar = psi(ts.sample.x, hm);
			double[] xiP = positiveBatchPsi[s]; // do not change
			double[] xiM = negativeBatchPsi[s];// do not change


			if(ts.label == 1) {
				double wx = linear.valueOf(w, xiPstar) - 1;
				if(wx > 0) {
					double e = eta/nb[0];
					for (int d = 0; d < w.length; d++) {
						w[d] += - xiPstar[d] * e;
					}
				}

				double e = nbd * eta / nb[0];
				
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

				e = eta * (1/(double)nb[0] + nbd/l.size());
				for (int d = 0; d < w.length; d++) {
					w[d] += e * xiP[d];
				}
			}
			else {
				double wx = -linear.valueOf(w, xiMstar) - 1;
				if(wx > 0) {
					double e = eta/nb[1];
					for (int d = 0; d < w.length; d++) {
						w[d] += e * xiMstar[d];
					}
				}

				double e = nbd * eta / nb[1];
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

				e = eta * (1/(double)nb[1] + nbd/l.size());
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
	
	protected void learnSGD(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l) {


		double newPrimal_Objectif = getPrimalObjective(l);
		double oldPrimal_Objectif = newPrimal_Objectif;
		
		int iter = 0;
		double[] lastW = new double[dim];
		
		do {
			iter +=1;
			oldPrimal_Objectif = newPrimal_Objectif;
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else {
				System.out.print(".");
			}
			System.arraycopy(w, 0, lastW, 0, dim);
			
			int e = 0;
			double[][] positiveBatchPsi = new double[l.size()][dim];		
			double[][] negativeBatchPsi = new double[l.size()][dim];		
			
			double maxw = 1.0 / Math.sqrt(lambda);
			double typw = Math.sqrt(maxw);
			double eta0 = typw;
			t = (long) (1 / (eta0 * lambda));
			
			//copy the v(w) part
			
			
			for(; e<maxEpochs; e++) {
				trainOnceEpochsSGD(l, positiveBatchPsi, negativeBatchPsi);
			}
			// update latent variables
			optimizeLatent(l);
			newPrimal_Objectif = getPrimalObjective(l);
			System.out.println("iter: "+iter+" obj"+newPrimal_Objectif);
		}while((oldPrimal_Objectif - newPrimal_Objectif> epsilon || iter < minCCCPIter) && (iter < maxCCCPIter));
//		while(iter < maxCCCPIter);
		System.out.format("total iteration %d times",iter);
//		System.arraycopy(lastW, 0, w, 0, dim);
		optimizeLatent(l);

		if(verbose == 0) {
			System.out.println("*");
		}
	}

	
public void trainOnceEpochsSGD(List<TrainingSample<LatentRepresentationSymil<X, H, H>>> l, double[][] positiveBatchPsi, double[][] negativeBatchPsi) {
		if(w == null)
			return;

		int imax = l.size();
		if(stochastic) {
//			Random seed = new Random(3);
//			Collections.shuffle(l, seed);
			Collections.shuffle(l);
		}

		
		for(int i=0; i<imax; i++) {
			//C/n
//			double eta = 1.0 / (lambda * t* (nb[0] + nb[1]));// learning rate
			double eta = 1.0 / (lambda * t);// learning rate
			//shrink w
			double s = 1 - eta * lambda; 
			for(int d=0; d<w.length; d++) {
				w[d] *= s;
			}
			
			TrainingSample<LatentRepresentationSymil<X, H, H>> ts = l.get(i);
			H[] hphn = optimizeH(ts.sample.x);
			H hp =hphn[0];
			H hn =hphn[1];
			
			double y = ts.label;
			double[] maxPsi = psi(ts.sample.x, hp);
			double[] minPsi = psi(ts.sample.x, hn);
			double maxValue =  linear.valueOf(w, maxPsi);
			double minValue =  linear.valueOf(w, minPsi);
			
			double[] xiP = psi(ts.sample.x, ts.sample.hp);
			double[] xiM = psi(ts.sample.x, ts.sample.hn);
//			if (i == 8){
//				System.out.println(positiveBatchPsi[8][5]);
//			}
			
			if (y==1){
				double e = ( 1 + nbd )  * eta;
				
				for(int d=0; d<dim; d++) {
					w[d] -= -  xiP[d] * e;
				}
				
				e  = 1  * eta ;
				if (maxValue >1){
					for(int d=0; d<dim; d++) {
						w[d] -=  maxPsi[d] * e ;
					}
				}
				
			    e = nbd * eta ;
				
				if (1-minValue > maxValue){
					for(int d=0; d<dim; d++) {
						w[d] -= -minPsi[d] *e;
					}
				}
				else{
					for(int d=0; d<dim; d++) {
						w[d] -= maxPsi[d] *e;
					}
				}
			}
	
			else if (y == -1){
				
				double e = eta * ( 1  + nbd  );
				for(int d=0; d<dim; d++) {
					w[d] -= xiM[d] * e;
				}
			
				
				e = eta;
				
				if(-minValue>1 ){
					for(int d=0; d<dim; d++) {
						w[d] -= - minPsi[d] * e;
					}
				}
				
				e = nbd * eta  ;
				
				if (1+maxValue > - minValue){
					for(int d=0; d<dim; d++) {
						w[d] -= maxPsi[d] *e;
					}
				}
				
				else{
					for(int d=0; d<dim; d++) {
						w[d] -= -minPsi[d] *e;
					}
				}
			}
//			if (y==1){
//				double e = ( (1/(double)nb[0]) + (nbd/l.size()) )  * eta;
//				
//				for(int d=0; d<dim; d++) {
//					w[d] -= -  positiveBatchPsi[i][d] * e;
//				}
//				
//				e  = ( 1 / (double)nb[0]) * eta ;
//				if (maxValue >1){
//					for(int d=0; d<dim; d++) {
//						w[d] -=  maxPsi[d] * e ;
//					}
//				}
//				
//			    e = nbd * eta / nb[0] ;
//				
//				if (1-minValue > maxValue){
//					for(int d=0; d<dim; d++) {
//						w[d] -= -minPsi[d] *e;
//					}
//				}
//				else{
//					for(int d=0; d<dim; d++) {
//						w[d] -= maxPsi[d] *e;
//					}
//				}
//			}
//	
//			else if (y == -1){
//				
//				double e = eta * ( (1/(double)nb[1]) + (nbd / l.size()) );
//				for(int d=0; d<dim; d++) {
//					w[d] -= negativeBatchPsi[i][d] * e;
//				}
//			
//				
//				e = eta/nb[1];
//				
//				if(-minValue>1 ){
//					for(int d=0; d<dim; d++) {
//						w[d] -= - minPsi[d] * e;
//					}
//				}
//				
//				e = nbd * eta / nb[1] ;
//				
//				if (1+maxValue > - minValue){
//					for(int d=0; d<dim; d++) {
//						w[d] -= maxPsi[d] *e;
//					}
//				}
//				
//				else{
//					for(int d=0; d<dim; d++) {
//						w[d] -= -minPsi[d] *e;
//					}
//				}
//			}
			t += 1;			

		}

	}

	@Override
	protected void showParameters() {
		System.out.println("maxCCCPIter= " + maxCCCPIter + "\t maxEpochs= " + maxEpochs +
				"\tstochastic= " + stochastic +
				"\t\t" + maxEpochs*maxCCCPIter);
		if(optim == 1) {
			System.out.println("Pegasos");
		}
	}

	public int getMaxCCCPIter() {
		return maxCCCPIter;
	}

	public void setMaxCCCPIter(int maxCCCPIter) {
		this.maxCCCPIter = maxCCCPIter;
	}


	public int getMaxEpochs() {
		return maxEpochs;
	}

	public void setMaxEpochs(int maxEpochs) {
		this.maxEpochs = maxEpochs;
	}

	public int getOptim() {
		return optim;
	}

	public void setOptim(int optim) {
		this.optim = optim;
	}

	/**
	 * @return the stochastic
	 */
	public boolean isStochastic() {
		return stochastic;
	}

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
	
	public int getMaxEpoch() {
		return maxEpochs;
	}

	public void setMaxEpoch(int maxEpochs) {
		this.maxEpochs = maxEpochs;
	}
	
	public String getGazeType() {
		return gazeType;
	}
	public void setGazeType(String gazeType) {
		this.gazeType = gazeType;
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
	
	public void setNbd(double nbd){
		this.nbd = nbd;
	}
	public double getNbd(){
		return nbd;
	}
	
	
	

}
