package fr.durandt.jstruct.ssvm.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.struct.StructuralClassifier;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * <b>Multiclass Latent Locally Linear Support Vector Machines</b><br />
 * Marco Fornoni, Barbara Caputo, and Francesco Orabona <br />
 * ACML 2013 <br />
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class ML3 implements StructuralClassifier<double[], Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4405221301074540360L;

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


	protected abstract void learning(List<STrainingSample<double[], Integer>> l);


	@Override
	public Integer prediction(double[] x) {
		int ypredict = -1;
		double max = -Double.MAX_VALUE;
		// For each class, compute the score
		for(int y : listClass) {
			// Compute the score for given class y
			double val = valueOf(x, y);
			if(val > max) {
				max = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	@Override
	public void train(List<STrainingSample<double[], Integer>> l) {

		// Initialize the dimension of the features
		d = l.get(0).input.length;

		// Search the number of classes
		c = 0;
		for(STrainingSample<double[], Integer> ts : l) {
			c = Math.max(c, ts.output);
		}
		c++;
		listClass = new ArrayList<Integer>();
		for(int i=0; i<c; i++) {
			listClass.add(i);
		}
		double[] nb = new double[c];
		for(STrainingSample<double[], Integer> ts : l) {
			nb[ts.output]++;
		}

		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train ML3 \tlambda: " + lambda + "\tdim: " + d + "\tsub-models= " + m + "\tp= " + p);
		System.out.println("Multiclass - classes: " + listClass + "\t" + Arrays.toString(nb));
		showParametersLearning();
		System.out.println("----------------------------------------------------------------------------------------");

		// Initialize w
		w = computeWinit();

		long startTime = System.currentTimeMillis();
		learning(l);
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("dim: " + d + "\tsub-models= " + m + "\tp= " + p);
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
	protected double[] computeOptimalBeta(double[][][] w, double[] x, int y, double p) {

		// compute c+
		double[] cplus = new double[w[y].length];
		for(int i=0; i<cplus.length; i++) {
			cplus[i] = Math.max(0, VectorOperations.dot(w[y][i], x));
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
	 * @param beta
	 * @param y
	 * @return beta^T W_y x
	 */
	protected double valueOf(double[] x, double[] beta, int y) {
		double[] tmp = new double[beta.length];
		for(int i=0; i<tmp.length; i++) {
			tmp[i] = VectorOperations.dot(w[y][i], x);
		}
		return VectorOperations.dot(tmp, beta);
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
					winit[i][j][k] = Math.random();
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
	 * argmax_{y != yi}
	 * @param x
	 * @param yi
	 * @return
	 */
	public Integer prediction(double[] x, int yi) {
		int ypredict = -1;
		double max = -Double.MAX_VALUE;
		for(int y : listClass) {
			if(y != yi) {
				double val = valueOf(x, y);
				if(val > max) {
					max = val;
					ypredict = y;
				}
			}
		}
		return ypredict;
	}

	/**
	 * Compute the score for a given class y
	 * @param x
	 * @param y
	 * @return
	 */
	public double valueOf(double[] x, int y, double[][][] w) {
		if(verbose > 2) {
			System.out.println("y= " + y + "\t" + w[y].length);
		}
		if(p > 1) {
			double[] c = new double[w[y].length];
			for(int i=0; i<c.length; i++) {
				c[i] = Math.max(0, VectorOperations.dot(w[y][i], x));
			}
			double q = (double)p/(double)(p-1.);
			return VectorOp.pnorm(c, q);
		}
		else {
			double[] beta = computeOptimalBeta(w, x, y, p);
			double val = valueOf(x, beta, y);
			return val;
		}
	}

	/**
	 * Compute the score for a given class y
	 * @param x
	 * @param y
	 * @param w
	 * @return
	 */
	public double valueOf(double[] x, int y) {
		return valueOf(x,y,w);
	}

	public double evaluation(List<STrainingSample<double[], Integer>> l) {
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<double[], Integer> ts : l){
			int ypredict = prediction(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}


	protected double loss(List<STrainingSample<double[], Integer>> l) {
		double loss = 0;
		for(STrainingSample<double[], Integer> ts : l) {
			int ybar = prediction(ts.input, ts.output);
			loss += Math.max(0, 1 + valueOf(ts.input, ybar) - valueOf(ts.input, ts.output));
		}
		loss /= l.size();
		return loss;	
	}

	protected double primalObj(List<STrainingSample<double[], Integer>> l) {
		double obj = 0.5 * lambda * Math.pow(frobenius(w),2);
		double loss = loss(l);
		System.out.println("lambda/2*||w||_F^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}

	public String toString() {
		return "ml3_lambda_" + lambda + "_m_" + m + "_p_" + p;
	}
	protected void showParametersLearning() {};

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
		if(p<1) {
			p=1.;
			System.out.println("ERROR p must be >=1.\n p = 1" );
		}
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
