package fr.durandt.jstruct.latent.lssvm.multiclass;

import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.LatentStructuralClassifier;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LatentML3<X,H> implements LatentStructuralClassifier<X,Integer,H> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8963403215286979225L;

	// c x m x d
	protected double[][][] w = null;

	// p-norm 1 <= p < infinity
	protected double p = 1.;
	// regularization parameter
	protected double lambda = 1e-4;

	// number of models per class
	protected int m = 5;
	// dimension of the feature space
	protected int d = 0;
	// number of classes
	protected int c = 0;

	// list of classes {0,1,...,c-1}
	protected List<Integer> listClass = null;

	protected double tau = 1.;

	protected int verbose = 0;


	//////////////////////////////////////////////////////////////////////////////////////////////////////

	protected abstract double[] psi(X x, H h);
	protected abstract H prediction(X x, Integer y, double[][][] w);
	protected abstract void init(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l);
	protected abstract void learn(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l);

	//////////////////////////////////////////////////////////////////////////////////////////////////////

	@Override
	public Integer prediction(LatentRepresentation<X,H> lr) {
		return prediction(lr.x, w);
	}

	protected Integer prediction(X x, double[][][] w) {
		int ypredict = -1;
		double max = -Double.MAX_VALUE;
		// For each class, compute the score
		for(int y : listClass) {
			// Compute the score for given class y
			double val = valueOf(x, y, w);
			if(verbose>2) {
				System.out.println("y= " + y + "\t" + val);
			}
			if(val > max) {
				max = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	@Override
	public void train(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {

		init(l);

		double[] nb = new double[c];
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l) {
			nb[ts.output]++;
		}

		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train Latent ML3 \tlambda: " + lambda + "\tdim: " + d + "\tmodels= " + m + "\tp= " + p);
		System.out.println("SSVM multiclass - classes: " + listClass + "\t" + Arrays.toString(nb));
		System.out.println("----------------------------------------------------------------------------------------");

		// Initialize w
		w = computeWinit();

		long startTime = System.currentTimeMillis();
		learn(l);
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("dim: " + d + "\tmodels= " + m + "\tp= " + p);
	}


	/**
	 * Eq. (11) 
	 * 
	 * @param w
	 * @param x
	 * @param y
	 * @param p >= 1
	 * @return
	 */
	protected double[] computeOptimalBeta(double[][][] w, X x, int y, H h, double p) {

		// compute c+
		double[] cplus = new double[w[y].length];
		for(int i=0; i<cplus.length; i++) {
			cplus[i] = Math.max(0, VectorOperations.dot(w[y][i], psi(x,h)));
		}

		double[] betaStar = new double[cplus.length];

		if(p>1) {
			// Compute beta^* with Eq. (14) 
			// Closed form solution
			double q = p/(p-1.);
			// Compute the q-norm of c^+
			double qnorm = VectorOp.pnorm(cplus, q);
			if(qnorm > 0) {
				for(int i=0; i<cplus.length; i++) {
					betaStar[i] = Math.pow(cplus[i]/qnorm, q-1.);
				}
			}
			else {
				betaStar = cplus;
			}
		}
		else if(p == 1) {
			// compute beta^* with Eq. (15)
			double max = 0; //-Double.MAX_VALUE;
			int index = -1;
			for(int i=0; i<cplus.length; i++) {
				if(cplus[i] > max) {
					max = cplus[i];
					index = i;
				}
			}
			//betaStar[index] = 1;
			if(index >= 0) {
				betaStar[index] = 1;
			}
		}
		else {
			System.out.println("Value of p non supported (p >= 1)");
			System.exit(0);
		}
		return betaStar;
	}

	/**
	 * Compute the score for a given beta and class y
	 * @param x
	 * @param y
	 * @param h
	 * @param beta
	 * @return beta^T W_y psi(x,h)
	 */
	protected double valueOf(X x, int y, H h, double[] beta, double[][][] w) {
		double[] tmp = new double[beta.length];
		for(int i=0; i<tmp.length; i++) {
			tmp[i] = VectorOperations.dot(w[y][i], psi(x,h));
		}
		return VectorOperations.dot(tmp, beta);
	}

	protected double valueOf(X x, Integer y, H h, double[] beta) {
		return valueOf(x, y, h, beta, w);
	}

	/**
	 * Compute the initial w
	 * @return
	 */
	protected double[][][] computeWinit() {
		double[][][] winit = new double[c][m][d];
		for(int i=0; i<c; i++) {
			for(int j=0; j<m; j++) {
				for(int k=0; k<d; k++) {
					winit[i][j][k] = 0.1*Math.random();
				}
			}
		}
		return winit;
	}

	/**
	 * Compute the Frobenius norm of matrix w
	 * @param w
	 * @return Frobenius norm of matrix w
	 */
	protected double frobenius(double[][][] w) {
		double norm = 0;
		for(int i=0; i<w.length; i++) {
			for(int j=0; j<w[i].length; j++) {
				for(int k=0; k<w[i][j].length; k++) {
					norm += w[i][j][k]*w[i][j][k];
				}
			}
		}
		return Math.sqrt(norm);
	}

	/**
	 * Compute beta^T * x
	 * @param beta
	 * @param x
	 * @return
	 */
	protected double[][] computeMatrix(double[] beta, double[] x) {
		double[][] mat = new double[beta.length][x.length];
		for(int i=0; i<mat.length; i++) {
			for(int j=0; j<mat[i].length; j++) {
				mat[i][j] = beta[i]*x[j];
			}
		}
		return mat;
	}

	/**
	 * Compute the score for a given class y
	 * @param x
	 * @param y
	 * @param h
	 * @param w
	 * @return beta^T W_y psi(x,h)
	 */
	public double valueOf(X x, int y, H h, double[][][] w) {
		if(p > 1) {
			double[] c = new double[w[y].length];
			for(int i=0; i<c.length; i++) {
				c[i] = Math.max(0, VectorOperations.dot(w[y][i], psi(x,h)));
				if(verbose>2) {
					System.out.println("i= " + i + "\tc= " + VectorOperations.dot(w[y][i], psi(x,h)));
				}
			}
			double q = (double)p/(double)(p-1.);
			if(verbose>2) {
				System.out.println("y= " + y + "\th= " + h + "\tq= " + q + "\tc= " + Arrays.toString(c));
			}
			return VectorOp.pnorm(c, q);
		}
		else {
			double[] beta = computeOptimalBeta(w, x, y, h, p);
			double val = valueOf(x, y, h, beta, w);
			return val;
		}
	}

	/**
	 * Compute the score for a given class y
	 * @param x
	 * @param y
	 * @return
	 */
	protected double valueOf(X x, int y) {
		return valueOf(x, y, w);
	}

	protected double valueOf(X x, int y, double[][][] w) {
		// Compute the best latent variable for class y
		H h = prediction(x, y, w);
		// Compute the score for the best latent variable for class y
		return valueOf(x, y, h, w);
	}

	/**
	 * Compute the latent variable inference 
	 * @param x
	 * @param y
	 * @return
	 */
	protected H prediction(X x, Integer y) {
		return prediction(x,y,w);
	}

	/**
	 * argmax_{y != yi}
	 * @param x
	 * @param yi
	 * @return
	 */
	protected Integer predictionLAI(STrainingSample<LatentRepresentation<X,H>,Integer> ts, double[][][] w) {
		int ypredict = -1;
		double max = -Double.MAX_VALUE;
		for(int y : listClass) {
			if(y != ts.output) {
				double val = valueOf(ts.input.x, y, w);
				if(val > max) {
					max = val;
					ypredict = y;
				}
			}
		}
		return ypredict;
	}

	protected Integer predictionLAI(STrainingSample<LatentRepresentation<X,H>,Integer> ts) {
		return predictionLAI(ts, w);
	}

	public double evaluation(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		double accuracy = 0;
		int nb = 0;
		verbose = 1;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l){
			int ypredict = prediction(ts.input);
			//System.out.println("yi= " + ts.output + "\typredict= " + ypredict);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		verbose = 1;
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}


	protected double loss(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		double loss = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l) {
			int ybar = predictionLAI(ts);
			loss += Math.max(0, 1 + valueOf(ts.input.x, ybar) - valueOf(ts.input.x, ts.output));
		}
		loss /= l.size();
		return loss;	
	}

	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		double obj = 0.5 * lambda * Math.pow(frobenius(w),2);
		double loss = loss(l);
		System.out.println("lambda/2*||w||_F^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}

	public String toString() {
		return "latentml3_lambda_" + lambda + "_m_" + m + "_p_" + p;
	}

	public double[][][] getW() {
		return w;
	}

	public void setW(double[][][] w) {
		this.w = w;
	}

	public double getP() {
		return p;
	}

	public void setP(double p) {
		this.p = p;
	}

	public double getLambda() {
		return lambda;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public int getM() {
		return m;
	}

	public void setM(int m) {
		this.m = m;
	}

	/**
	 * @return the verbose
	 */
	public int getVerbose() {
		return verbose;
	}

	/**
	 * @param verbose the verbose to set
	 */
	public void setVerbose(int verbose) {
		this.verbose = verbose;
	}


}
