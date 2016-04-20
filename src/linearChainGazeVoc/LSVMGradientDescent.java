/**
 * 
 */
package linearChainGazeVoc;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LSVMGradientDescent<X,H> extends LSVM<X,H> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -667731866975742617L;


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
	protected HashMap<String , Integer> groundTruthGazeMap = new HashMap<String , Integer>();
	
	private long t=0;
	
	abstract public double[] getGroundGazePsi(TrainingSample<LatentRepresentation<X,H>> ts);
	abstract public HashMap<String , Integer> GroundTruthGazeRegion(List<TrainingSample<LatentRepresentation<BagImage, Integer>>> l);
	abstract public Object[] LAI(TrainingSample<LatentRepresentation<X, H>> ts) ;

	
	@Override
	protected void learn(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		if(optim == 2) {
			mainTrainingLoop(l, null);
		}
	}
	
	@Override
	protected void learn(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut) {
		if(optim == 2) {
			mainTrainingLoop(l, trainingDetailFileOut);
		}
	}

	protected void mainTrainingLoop(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut) {

		// Shift t in order to have a
		// reasonable initial learning rate.
		// This assumes |x| \approx 1.
		double maxw = 1.0 / Math.sqrt(lambda);
		double typw = Math.sqrt(maxw);
		double eta0 = typw;
		t = (long) (1 / (eta0 * lambda));
		
		/*
		 * double newPrimal_Objectif = getPrimalObjective(l, trainingDetailFileOut);
		double oldPrimal_Objectif = newPrimal_Objectif;
		*/
		
		int iter = 0;
		double[] lastWl = new double[dim];
		for (int innerloop=0;innerloop<10;innerloop++){
//		do {
			System.out.println(getPrimalObjective(l, trainingDetailFileOut));
			iter +=1;
			//oldPrimal_Objectif = newPrimal_Objectif;
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else {
				System.out.print(".");
			}
			//System.out.println("objectif: "+oldPrimal_Objectif);
			System.arraycopy(wl, 0, lastWl, 0, dim);
			
			int e = 0;
			for(; e<maxEpochs; e++) {
				trainOnceEpochsSGDWL(l);
//				System.out.println(" after a loop of wl:"+getPrimalObjective(l, trainingDetailFileOut));
			}
			// update all latent variables by new wl
			optimizeLatent(l);
			
			trainWC(l);
			
			System.out.println("innerloop:"+innerloop);
//			newPrimal_Objectif = getPrimalObjective(l, trainingDetailFileOut);
		}
//		while((oldPrimal_Objectif - newPrimal_Objectif> epsilon || iter < minCCCPIter) && (iter < maxCCCPIter));
		
		try {
			trainingDetailFileOut.write("total_iteration_time:"+iter);
			trainingDetailFileOut.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.format("total iteration %d times",iter);
		System.arraycopy(lastWl, 0, wl, 0, dim);
		optimizeLatent(l);
		
		if(verbose == 0) {
			System.out.println("*");
		}
	}

	
	/**
	 * Update the separating hyperplane by learning one epoch on given training list
	 * @param l the training list
	 */
	
	
	
	
	public void trainOnceEpochsSGDWL(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		
		
		int imax = l.size();
		
		if(stochastic) {
			Collections.shuffle(l);
		}
		
		//psi of cp2 term
		/*double[][] cp2BatchPsi = new double[l.size()][dim];		
		for (int i=0; i<l.size();i++){
			if (l.get(i).label == 1){
				double[] cp2Psi = psi(l.get(i).sample.x, l.get(i).sample.h);
				for (int j=0; j<dim;j++){
					cp2BatchPsi[i][j]=cp2Psi[j];
				}
			}
		}*/
		
		for(int i=0; i<imax; i++) {
			
			double eta = 1.0 / (lambda * t);// learning rate
			
			double s = 1 - eta * lambda; // shrink
			//shrink w
			for(int d=0; d<wl.length; d++) {
				wl[d] *= s;
			}
			
			TrainingSample<LatentRepresentation<X,H>> ts = l.get(i);
			Object[] lai = LAI(ts);
			H laiRegion = (H)lai[0];
			double laiValue = (double)lai[1];
			//wl gradient
			double[] gazePsi = getGroundGazePsi(ts);
			for(int d=0; d<dim; d++) {
				wl[d] -= eta *  (psi(ts.sample.x, laiRegion)[d] - gazePsi[d]);
			}
			
//			double[] innerX = psi(ts.sample.x, ts.sample.h);
//			double y = ts.label;
//			double z = y * linear.valueOf(w, innerX);
//			
//			if (z < 1 && y ==-1) {
//				for(int d=0; d<dim; d++) {
//					w[d] -= innerX[d] * eta;
//				}
//			}
//			else if (z >= 1 && y ==-1){
//				
//			}
//			else if (z < 1 && y == 1){
//				for(int d=0; d<dim; d++) {
//					w[d] -= (-cp2BatchPsi[i][d]) * eta;
//				}
//			}
//			else if (z >= 1 && y == 1){
//				for(int d=0; d<dim; d++) {
//					w[d] -= (innerX[d]-cp2BatchPsi[i][d]) * eta;
//				}
//			}
			t += 1;
		}
	}
	public void trainWC(List<TrainingSample<LatentRepresentation<X,H>>> l){
		int totalExampleNum = nb[0]+nb[1];
		double[] labels =new double[totalExampleNum];
		FeatureNode[][] trainingSetFeatures = new FeatureNode[totalExampleNum][dim];
		
		// quadratic
//		FeatureNode[] tp1 = {new FeatureNode(1,  2), new FeatureNode(2,   4)};

//		FeatureNode[][] trainingSetWithUnknown = {
//		    tp1, tp2,  tp3, tp4, tp5, tp6, tp7, tp8
//		};
		
		int cnt = 0;
		for (TrainingSample<LatentRepresentation<X,H>> ts: l){
			labels[cnt] = ts.label;
			FeatureNode[] tp = new FeatureNode[dim];
			for (int findex=0; findex<dim; findex++){
				tp[findex]  = new FeatureNode(findex+1, psi(ts.sample.x, ts.sample.h)[findex]);
			}
			trainingSetFeatures[cnt] = tp;
			cnt+=1;
		};
		
	    Problem problem = new Problem();
	    problem.l = totalExampleNum;
	    problem.n = dim;
	    problem.x = trainingSetFeatures;
	    problem.y = labels;

	    SolverType solver = SolverType.L2R_L1LOSS_SVC_DUAL; // -s 0
	    double C = 1e4; // cost of constraints violation
	    double eps = 0.001; // stopping criteria

	    Parameter parameter = new Parameter(solver, C, eps);
	    Model model = Linear.train(problem, parameter);
	    wc = model.getFeatureWeights();
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
	
	public HashMap<String, Integer> getGroundTruthGazeMap() {
		return groundTruthGazeMap;
	}
	public void setGroundTruthGazeMap(HashMap<String, Integer> groundTruthGazeMap) {
		this.groundTruthGazeMap = groundTruthGazeMap;
	}


}
